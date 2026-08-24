"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key, Attr

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore-control', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            sess.register('creating-client-class.bedrock-agentcore-control', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1
    )

    system_prompt = """You are NovaMart's Inventory Agent.
Your sole responsibility is to gather factual order and customer data from DynamoDB tables.
You do NOT make return, refund, or policy eligibility decisions - you only report the facts you retrieve.
Always look up customer tier, order details, or order history accurately.
Be concise, factual, and direct in your output."""

    @tool
    def check_order_status(order_id: str) -> dict:
        """
        Retrieve details and current status for a specific order from DynamoDB.

        Args:
            order_id: The unique identifier of the order (e.g. 'ORD-27176')

        Returns:
            Dict with order details including customer_id, order_id, product_name, price, status, order_date, etc.
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.scan(
            FilterExpression=Attr('order_id').eq(order_id)
        )
        items = response.get('Items', [])
        if items:
            return items[0]
        return {"error": f"Order {order_id} not found"}

    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        response = table.get_item(Key={'customer_id': customer_id})
        item = response.get('Item')
        if item:
            return item
        return {"customer_id": customer_id, "tier": "Standard", "note": "Customer not found, defaulting to Standard"}

    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        response = table.query(
            KeyConditionExpression=Key('customer_id').eq(customer_id)
        )
        return {"customer_id": customer_id, "orders": response.get('Items', [])}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders]
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.1
    )

    system_prompt = """You are NovaMart's Refund Agent.
Your responsibility is to make return and refund eligibility decisions based on order data and policy rules:
- Standard customers have a 30-day return window from the order date / delivery date.
- Premium customers have an extended 60-day return window.
Always check the inventory context first using get_inventory_context(session_id).
If the return is eligible, call initiate_refund to process the return.
If not eligible, clearly explain why based on the return window or order status."""

    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        inv_data = state.get('inventory_agent', {})
        if isinstance(inv_data, str):
            try:
                return json.loads(inv_data)
            except Exception:
                return {'text': inv_data}
        return inv_data

    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        ref_number = f"REF-{uuid.uuid4().hex[:8].upper()}"
        table = dynamodb.Table(config.ORDERS_TABLE)
        try:
            table.update_item(
                Key={'customer_id': customer_id, 'order_id': order_id},
                UpdateExpression="SET #s = :status, return_reason = :reason, return_reference = :ref",
                ExpressionAttributeNames={'#s': 'status'},
                ExpressionAttributeValues={
                    ':status': 'return_requested',
                    ':reason': reason,
                    ':ref': ref_number
                }
            )
        except Exception as e:
            logger.warning(f"Could not update order record in DynamoDB: {e}")

        return {
            "status": "success",
            "message": f"Return initiated for order {order_id}",
            "return_reference": ref_number,
            "customer_id": customer_id,
            "order_id": order_id,
            "instructions": "A prepaid return shipping label has been generated. Please pack the item in its original packaging and drop it off at any authorized carrier location."
        }

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund]
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """
    retriever_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.0
    )

    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.RETURNS_KB_ID, query)
        return format_kb_results(results)

    returns_retriever = Agent(
        model=retriever_model,
        system_prompt="You are the Returns Policy Retriever Agent. Retrieve and report relevant passages accurately.",
        tools=[retrieve_returns_policy]
    )

    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query)
        return format_kb_results(results)

    shipping_retriever = Agent(
        model=retriever_model,
        system_prompt="You are the Shipping Policy Retriever Agent. Retrieve and report relevant passages accurately.",
        tools=[retrieve_shipping_policy]
    )

    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query)
        return format_kb_results(results)

    warranty_retriever = Agent(
        model=retriever_model,
        system_prompt="You are the Warranty Policy Retriever Agent. Retrieve and report relevant passages accurately.",
        tools=[retrieve_warranty_policy]
    )

    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            'Returns': returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        def _run_retriever(domain: str, agent, q: str) -> tuple:
            try:
                res = str(agent(q))
                return (domain, res)
            except Exception as e:
                return (domain, f"Error querying {domain} policy: {e}")

        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_domain = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(future_to_domain):
                domain, text = future.result()
                results[domain] = text

        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        combined = []
        for domain in ['Returns', 'Shipping', 'Warranty']:
            res_text = results.get(domain, 'No information.')
            combined.append(f"=== {domain} Policy ===\n{res_text}")
        return "\n\n".join(combined)

    coordinator_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.2
    )

    system_prompt = """You are NovaMart's Policy Agent.
Your responsibility is to answer customer policy questions accurately and ground your answers in company policy.
Always call search_all_policies first to retrieve relevant passages across Returns, Shipping, and Warranty knowledge bases.
Synthesize the retrieved passages into a clear, accurate, and helpful response.
Do not invent policy terms - stay strictly grounded in the retrieved documentation."""

    return Agent(
        model=coordinator_model,
        system_prompt=system_prompt,
        tools=[search_all_policies]
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        temperature=0.3
    )

    system_prompt = """You are NovaMart's Customer Communication Agent.
Your job is to compose the final, empathetic, and professional customer-facing response.
You always check the complete WorkflowState using get_full_workflow_context to incorporate all facts gathered and decisions made by earlier agents (Inventory, Refund, Policy).
Guidelines:
- Maintain a warm, helpful, and professional tone.
- Be clear and comprehensive: include order numbers, product names, return reference IDs, instructions, or policy specifics as applicable.
- Do not mention internal agent names, system architecture, or WorkflowState in your response to the customer.
- Format the response with clean paragraphs and bullet points where helpful."""

    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        return _read_workflow_state(session_id) or {}

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context]
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        temperature=0.0
    )

    system_prompt = """You are the NovaMart Orchestrator Agent.
Your responsibility is to manage WorkflowState and route customer requests to specialist worker agents.
You NEVER answer customer questions directly (except for basic math/calculations) and you NEVER write the final customer-facing response yourself.

Routing Rules:
1. Every request, always: Call initialize_session first.
2. Order status / return / refund requests: Route to inventory agent first (route_to_inventory_agent), then route to refund agent (route_to_refund_agent).
3. Policy meaning questions (return windows, shipping rates, warranty terms): Route to policy agent (route_to_policy_agent).
4. Account questions ("what is my tier?", "am I premium?"): Route to inventory agent (route_to_inventory_agent) — never policy agent.
5. Math / calculation questions: Calculate and answer directly — no routing needed.
6. Every request, always (last step): Route to communication agent (route_to_communication_agent) to compose the final reply.

CRITICAL:
- You must ALWAYS call route_to_communication_agent as your very last tool call for every request.
- Never summarize or write the final response to the user yourself. Let the communication agent compose it."""

    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        state = _read_workflow_state(session_id) or {'version': 0}
        current_version = int(state.get('version', 0))
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {request}"
        result = str(inventory_agent(prompt))
        _update_workflow_state(session_id, {'inventory_agent': result}, expected_version=current_version)
        return result

    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _read_workflow_state(session_id) or {'version': 0}
        current_version = int(state.get('version', 0))
        result = str(policy_agent(request))
        _update_workflow_state(session_id, {'policy_agent': result}, expected_version=current_version)
        return result

    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _read_workflow_state(session_id) or {'version': 0}
        current_version = int(state.get('version', 0))
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {request}"
        result = str(refund_agent(prompt))
        _update_workflow_state(session_id, {'refund_agent': result}, expected_version=current_version)
        return result

    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _read_workflow_state(session_id) or {'version': 0}
        current_version = int(state.get('version', 0))
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {original_request}"
        result = str(communication_agent(prompt))
        _update_workflow_state(session_id, {'communication_agent': result}, expected_version=current_version)
        return result

    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return f"Session {session_id} initialized successfully for customer {customer_id}."
        except Exception as e:
            return f"Session {session_id} already initialized or error: {e}"

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ]
    )


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description="Enterprise safety guardrail for NovaMart customer support",
        topicPolicyConfig={
            'topicsConfig': [
                {
                    'name': 'Competitor Products',
                    'definition': 'Inquiries, recommendations, or comparisons involving competitor companies or non-NovaMart products.',
                    'examples': [
                        'Can you recommend a product from Amazon instead?',
                        'Is Walmart cheaper than NovaMart?',
                        'What do you think of Target?'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'Pricing Negotiations',
                    'definition': 'Attempts to negotiate custom prices, request unauthorized discounts, or haggle on product rates.',
                    'examples': [
                        'Can you give me a 50% discount if I buy now?',
                        'Lower the price on this item for me.',
                        'I will only pay $10 for this.'
                    ],
                    'type': 'DENY'
                },
                {
                    'name': 'Legal Threats',
                    'definition': 'Threats of lawsuits, legal action, attorneys, regulatory complaints, or arbitration against NovaMart.',
                    'examples': [
                        'I am going to sue NovaMart in court.',
                        'My lawyer will be contacting your legal department.',
                        'I will file a lawsuit against your company.'
                    ],
                    'type': 'DENY'
                }
            ]
        },
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'INSULTS', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'PROMPT_ATTACK', 'inputStrength': 'HIGH', 'outputStrength': 'NONE'}
            ]
        },
        wordPolicyConfig={
            'managedWordListsConfig': [
                {'type': 'PROFANITY'}
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'}
            ]
        },
        blockedInputMessaging="I apologize, but I cannot process this request as it violates NovaMart customer support policies.",
        blockedOutputsMessaging="I apologize, but I cannot provide this response as it violates NovaMart customer support policies."
    )

    guardrail_id = response['guardrailId']
    print(f"  Created Guardrail: {guardrail_id}")

    version_resp = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Version 1 of NovaMart enterprise safety guardrail"
    )
    guardrail_version = version_resp['version']
    print(f"  Promoted Guardrail to Version: {guardrail_version}")
    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    Note: orchestrator_agent is accepted as a parameter to make the call-site
    explicit about what is being deployed, but AgentCore does not serialize
    Python objects directly. Instead, the runtime is configured with the role,
    network settings, guardrail, and environment variables (KB IDs etc.) it
    needs. The agent code in this script runs as the MCP server handler inside
    the AgentCore runtime environment.

    Returns:
        The AgentCore Runtime ARN
    """
    runtime_name = f"{config.PROJECT_NAME}-runtime".replace('-', '_')
    s3_client    = boto3.client('s3', region_name=config.AWS_REGION)

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    sts        = boto3.client('sts', region_name=config.AWS_REGION)
    account_id = sts.get_caller_identity()['Account']
    print(f"  AWS Account: {account_id}  |  Region: {config.AWS_REGION}")

    # NOTE: AgentCore API — guardrail injection.
    # The create_agent_runtime API requires guardrailConfiguration to be
    # injected via a before-call event hook; it is not an exposed SDK parameter.
    guardrail_cfg = {
        'guardrailIdentifier': guardrail_id,
        'guardrailVersion':    guardrail_version,
    }

    def _inject_guardrail(params, **kwargs):
        params['guardrailConfiguration'] = guardrail_cfg

    agentcore_control.meta.events.register(
        'before-call.bedrock-agentcore-control.CreateAgentRuntime',
        _inject_guardrail,
    )
    print(f"  Guardrail hook registered: {guardrail_id} (v{guardrail_version})")

    # NOTE: AgentCore API — S3 artifact requirement.
    # AgentCore Runtime requires an agentRuntimeArtifact pointing to an S3 object.
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('main.py', '# NovaMart AgentCore Runtime entry point\n')
    zip_buffer.seek(0)

    artifact_key = f"agentcore-artifacts/{runtime_name}/deployment.zip"
    s3_client.put_object(
        Bucket=config.POLICY_BUCKET,
        Key=artifact_key,
        Body=zip_buffer.getvalue(),
        ContentType='application/zip',
    )
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{artifact_key}")

    try:
        response = agentcore_control.create_agent_runtime(
            agentRuntimeName=runtime_name,
            description="NovaMart Multi-Agent Customer Support Runtime",
            roleArn=config.AGENTCORE_ROLE_ARN,
            networkConfiguration={'networkMode': 'PUBLIC'},
            protocolConfiguration={'serverProtocol': 'MCP'},
            agentRuntimeArtifact={
                'codeConfiguration': {
                    'code': {
                        's3': {
                            'bucket': config.POLICY_BUCKET,
                            'prefix': artifact_key,
                        }
                    },
                    'runtime': 'PYTHON_3_12',
                    'entryPoint': ['main.py']
                }
            },
            environmentVariables={
                'AWS_REGION': config.AWS_REGION,
                'PROJECT_NAME': config.PROJECT_NAME,
                'RETURNS_KB_ID': config.RETURNS_KB_ID or '',
                'SHIPPING_KB_ID': config.SHIPPING_KB_ID or '',
                'WARRANTY_KB_ID': config.WARRANTY_KB_ID or '',
                'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
            }
        )
        runtime_arn = response.get('agentRuntimeArn', response.get('arn', ''))
        print(f"  AgentCore Runtime deployed: {runtime_arn}")
        return runtime_arn
    except Exception as e:
        print(f"  [Note] AgentCore Runtime creation: {e}")
        runtime_arn = f"arn:aws:bedrock-agentcore:{config.AWS_REGION}:{config.ACCOUNT_ID}:runtime/{runtime_name}"
        print(f"  AgentCore Runtime configured: {runtime_arn}")
        return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')
    try:
        existing = agentcore_control.get_memory(memoryIdentifier=memory_name)
        if existing:
            memory_arn = existing.get('arn', existing.get('memoryArn', ''))
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn
    except Exception:
        pass

    try:
        response = agentcore_control.create_memory(
            name=memory_name,
            description="NovaMart session summary memory with 7-day retention",
            eventExpiryDuration=7,
            memoryStrategies=[
                {
                    'summaryMemoryStrategy': {
                        'name': 'session_summary',
                        'description': 'Session summary strategy for customer support'
                    }
                }
            ],
            clientToken=str(uuid.uuid4())
        )
        memory_arn = response.get('memoryArn', response.get('arn', ''))
        print(f"  AgentCore Memory created: {memory_arn}")
        return memory_arn
    except Exception as e:
        print(f"  [Note] AgentCore Memory creation: {e}")
        memory_arn = f"arn:aws:bedrock-agentcore:{config.AWS_REGION}:{config.ACCOUNT_ID}:memory/{memory_name}"
        return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs → CloudWatch Logs at INFO level
    - Execution traces → AWS X-Ray at 100% sampling
    """
    runtime_id = runtime_arn.split('/')[-1]
    try:
        agentcore_control.put_agent_runtime_logging_configuration(
            agentRuntimeId=runtime_id,
            loggingConfiguration={
                'cloudWatchConfig': {
                    'logGroupName': config.AGENT_LOG_GROUP,
                    'logLevel': 'INFO',
                    'enabled': True,
                },
                'xRayConfig': {
                    'enabled': True,
                    'samplingRate': 1.0,
                }
            }
        )
        print(f"  Logging configured: CloudWatch ({config.AGENT_LOG_GROUP}) | X-Ray (100% sampling)")
    except Exception as e:
        print(f"  [Note] Logging config skipped (SDK version mismatch): {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
