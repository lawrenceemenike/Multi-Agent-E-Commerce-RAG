# NovaMart: Enterprise Multi-Agent E-Commerce RAG System

An enterprise-grade, autonomous customer support AI layer built with the **Strands Agents SDK**, **Amazon Bedrock AgentCore**, **Amazon Bedrock Knowledge Bases**, and **DynamoDB**.

The system automates complex customer support operations — including order lookups, tier-based refund evaluations, multi-domain policy retrieval (Returns, Shipping, Warranty), and empathetic customer communication — while enforcing strict enterprise safety guardrails, session memory, and full observability.

---

## 🏗️ System Architecture

```
                                  Customer Request
                                         │
                                  OrchestratorAgent
                             (Claude 3 Haiku, Temp: 0.0)
                        [Initializes & Updates WorkflowState]
                                         │
         ┌───────────────────────────────┼───────────────────────────────┬───────────────────────────────┐
         ▼                               ▼                               ▼                               ▼
   InventoryAgent                   PolicyAgent                     RefundAgent                 CommunicationAgent
 (Sonnet, Temp: 0.1)             (Sonnet, Temp: 0.2)             (Sonnet, Temp: 0.1)            (Sonnet, Temp: 0.3)
   [DynamoDB Read]               [Multi-Agent RAG]              [Eligibility & Update]          [Final Empathy Reply]
         │                               │                               │                               │
         │                ┌──────────────┼──────────────┐                │                               │
         │                ▼              ▼              ▼                │                               │
         │         ReturnsRetriever ShippingRetriever WarrantyRetriever  │                               │
         │            (Temp: 0.0)      (Temp: 0.0)       (Temp: 0.0)     │                               │
         │           [Returns KB]     [Shipping KB]     [Warranty KB]    │                               │
         │                └─────── ThreadPoolExecutor ───┘               │                               │
         ▼                                                               ▼                               ▼
   Orders / Customers                                           Orders Table Update             Final Customer Response
     DynamoDB Tables                                              (Status -> REFUNDED)
```

---

## 🌟 Key Features & Agent Roles

### 1. OrchestratorAgent (Router & Session Manager)
- **Model:** Anthropic Claude 3 Haiku (`temperature=0.0`) for ultra-low latency, deterministic routing.
- **State Management:** Manages `WorkflowState` stored in DynamoDB with optimistic concurrency control (`version` attribute).
- **Routing Rules:**
  1. *Order Lookup* $\rightarrow$ `InventoryAgent`
  2. *Policy Questions* $\rightarrow$ `PolicyAgent`
  3. *Refund Requests* $\rightarrow$ `InventoryAgent` (for order context) $\rightarrow$ `RefundAgent`
  4. *Multi-Intent Requests* $\rightarrow$ Sequential specialist execution
  5. *Final Response Composition* $\rightarrow$ `CommunicationAgent`

### 2. Specialized Worker Agents
- **InventoryAgent (Claude 3 Sonnet, Temp: 0.1):** Queries DynamoDB tables (`OrdersTable`, `CustomersTable`) to retrieve order status, customer loyalty tiers, and past order history.
- **RefundAgent (Claude 3 Sonnet, Temp: 0.1):** Evaluates refund eligibility against customer tier policies (Standard: 30-day window, Premium: 60-day window) and updates the order status to `REFUNDED` in DynamoDB.
- **PolicyAgent & Retriever Sub-Agents (Parallel Multi-Agent RAG):**
  - Coordinates 3 domain-specialized retriever sub-agents: `ReturnsPolicyRetrieverAgent`, `ShippingPolicyRetrieverAgent`, and `WarrantyPolicyRetrieverAgent`.
  - Executes parallel queries across Bedrock Knowledge Bases using Python's `ThreadPoolExecutor` and synthesizes findings into a unified policy summary.
- **CommunicationAgent (Claude 3 Sonnet, Temp: 0.3):** Pulls the accumulated context from `WorkflowState` and composes a warm, empathetic, customer-ready message.

### 3. Enterprise Safety Guardrails
- **Bedrock Guardrail:** Enforces content moderation, topic denial (competitor comparisons, price negotiations), sensitive PII protection (blocking Credit Cards & SSNs; masking Emails & Phone Numbers), and profanity filtering.

### 4. Memory & Observability
- **AgentCore Memory:** Session-scoped memory using `SESSION_SUMMARY` strategy with 7-day retention.
- **CloudWatch Logging:** Structured application logs at `INFO` level under `/aws/bedrock/agentcore/udacity-agentcore`.
- **AWS X-Ray:** Distributed request tracing with 100% trace sampling.

---

## 📊 Verification & Test Results

```
───────────────────────────────────────────────────────
  Automated Test Results (python tests/test_agent.py all)
───────────────────────────────────────────────────────
  ✓ Task 2 - Multi-Agent Orchestration   [40/40 pts] (100%)
  ✓ Task 3 - Guardrails & Deployment     [20/20 pts] (100%)
  ✓ Task 4 - Memory Configuration        [15/15 pts] (100%)
  ✓ Task 5 - Bedrock Knowledge Bases     [25/25 pts] (100%)
  ✓ Task 6 - CloudWatch & X-Ray Tracing  [20/20 pts] (100%)
───────────────────────────────────────────────────────
  🎉 Final Score: 120/120 pts (100% PERFECT SCORE)
───────────────────────────────────────────────────────
```

---

## 📁 Repository Structure

```
├── diagrams/                           # Architectural diagrams and reference flows
│   ├── architecture-overview.png
│   ├── agents-and-tools-reference.png
│   └── request-flow-scenarios.png
├── infrastructure/
│   ├── starter_stack.yaml              # CloudFormation template for DynamoDB, S3, IAM, CloudWatch
│   └── seed_data.py                    # Script to populate mock customers, orders, and S3 policy docs
├── src/
│   ├── agent_orchestrator.py           # Core multi-agent implementation, guardrail & deployment logic
│   ├── bedrock_kb_retrieval.py         # Knowledge Base retrieval helper (Vector & Managed KBs)
│   ├── agent_utils.py                  # Agent runtime utilities and formatting helpers
│   └── demo.py                         # Interactive customer support CLI demo
├── tests/
│   └── test_agent.py                   # Comprehensive automated test suite
├── config.py                           # Central configuration loader
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment variable template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- AWS CLI configured with active credentials
- Amazon Bedrock model access enabled in `us-east-1`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/lawrenceemenike/Multi-Agent-E-Commerce-RAG.git
cd Multi-Agent-E-Commerce-RAG

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Deploy Foundation Infrastructure
```bash
# Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file infrastructure/starter_stack.yaml \
  --stack-name udacity-agentcore \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1

# Seed sample customers, orders, and S3 policy documents
python infrastructure/seed_data.py
```

### 4. Create Bedrock Knowledge Bases
Create 3 Managed Knowledge Bases in the AWS Bedrock Console pointing to your policy S3 bucket:
1. **Returns KB:** S3 Prefix `policies/returns/`
2. **Shipping KB:** S3 Prefix `policies/shipping/`
3. **Warranty KB:** S3 Prefix `policies/warranty/`

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Knowledge Base IDs:
```bash
cp .env.example .env
```
```ini
AWS_REGION=us-east-1
PROJECT_NAME=udacity-agentcore
RETURNS_KB_ID=your-returns-kb-id
SHIPPING_KB_ID=your-shipping-kb-id
WARRANTY_KB_ID=your-warranty-kb-id
```

### 6. Deploy & Run Automated Tests
```bash
# Deploy Guardrail and Runtime configuration
python src/agent_orchestrator.py deploy

# Run the complete test suite
python tests/test_agent.py all
```

---

## 📜 License
This project is licensed under the MIT License.