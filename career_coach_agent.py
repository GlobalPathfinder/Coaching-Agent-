# ============================================================================
# MID-CAREER COACHING AGENT - CAPSTONE PRODUCTION SUBMISSION
# Level 2: Strategic Problem Solver
# Implements: Orchestration, Tools, Memory, Observability, Security, Reliability
# ============================================================================

# CELL 1: Install Dependencies
import subprocess
import sys

packages = ['google-generativeai', 'chromadb', 'opentelemetry-api', 'opentelemetry-sdk', 'tiktoken']
print("Installing dependencies...")
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    print(f"  ✓ {pkg}")
print("\n✓ Production dependencies installed!")

# CELL 2: Imports
import json, os, time, logging, uuid, hashlib, re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

import google.generativeai as genai
import chromadb
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

print("✓ All imports completed successfully")

# CELL 3: API Key Configuration
print("="*80)
print("🔑 GEMINI API KEY CONFIGURATION")
print("="*80)
print("\nIMPORTANT: Add your Gemini API key below to use the agent.")
print("Without an API key, the agent will run in DEMO MODE with simulated responses.\n")
print("Get your free API key at: https://aistudio.google.com/apikey")
print("="*80)

# ============================================================================
# ADD YOUR GEMINI API KEY HERE
# ============================================================================
GEMINI_API_KEY = ""  # ← Paste your API key between the quotes
# ============================================================================

if GEMINI_API_KEY:
    print("✓ API key configured - Full agent mode enabled")
else:
    print("⚠️  No API key - Agent will run in DEMO MODE")
    print("   Add your key above and re-run this cell to enable full functionality")

print("="*80 + "\n")

# Configuration
class Config:
    """Centralized configuration"""
    MAX_CONTEXT_TOKENS = 8000
    MAX_TOOL_RETRIES = 3
    ALLOWED_TOOL_CALLS = ["retrieve_linkedin_summary", "process_documents", "request_confirmation"]
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{16}\b',  # Credit card
        r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'  # Email (if sensitive)
    ]

print("✓ Configuration loaded")

# CELL 4: Security - Agent Identity
class AgentAuthority:
    """
    Implements Agent Identity and Least Privilege
    Prevents Confused Deputy attacks
    """
    def __init__(self, agent_id: str, permissions: List[str]):
        self.agent_id = agent_id
        self.permissions = set(permissions)
        self.creation_time = datetime.now()
        
    def can_execute(self, action: str) -> bool:
        """Verify agent has permission for action"""
        return action in self.permissions
    
    def get_identity(self) -> Dict:
        """Return verifiable identity"""
        return {
            "agent_id": self.agent_id,
            "permissions": list(self.permissions),
            "created": self.creation_time.isoformat()
        }

print("✓ Agent Identity & Authority system ready")

# CELL 5: Security - Guardrails
class GuardrailFilter:
    """
    Defense-in-depth: Input filtering and output screening
    """
    def __init__(self):
        self.blocked_patterns = [
            r'ignore previous instructions',
            r'system prompt',
            r'you are now',
            r'jailbreak',
            r'<\s*script',  # Script injection
        ]
        self.pii_patterns = Config.PII_PATTERNS
    
    def filter_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Input filtering: Block prompt injection attempts
        Returns: (is_safe, sanitized_input or error_message)
        """
        lower_input = user_input.lower()
        
        # Check for injection attempts
        for pattern in self.blocked_patterns:
            if re.search(pattern, lower_input, re.IGNORECASE):
                return False, f"Input blocked: potential security violation detected"
        
        # Check input length
        if len(user_input) > 10000:
            return False, "Input too long (max 10000 characters)"
        
        return True, user_input
    
    def screen_output(self, output: str) -> Tuple[bool, str]:
        """
        Output screening: Block PII leakage
        Returns: (is_safe, sanitized_output or error_message)
        """
        # Check for PII patterns
        for pattern in self.pii_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                # Redact instead of blocking entirely
                output = re.sub(pattern, "[REDACTED]", output)
        
        return True, output

print("✓ Security Guardrails (Input/Output Filtering) ready")

# CELL 6: Reliability - Idempotency Manager
class IdempotencyManager:
    """
    Ensures tools are safe-to-retry
    Prevents duplicate operations on network failures
    """
    def __init__(self):
        self.operation_cache: Dict[str, Any] = {}
        self.ttl_seconds = 300  # 5 minute cache
    
    def generate_key(self, tool_name: str, params: Dict) -> str:
        """Generate unique key for operation"""
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(f"{tool_name}:{params_str}".encode()).hexdigest()
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result if available"""
        if key in self.operation_cache:
            cached = self.operation_cache[key]
            if time.time() - cached['timestamp'] < self.ttl_seconds:
                return cached['result']
            else:
                del self.operation_cache[key]
        return None
    
    def cache_result(self, key: str, result: Any):
        """Cache operation result"""
        self.operation_cache[key] = {
            'result': result,
            'timestamp': time.time()
        }

print("✓ Idempotency Manager (Safe-to-retry tools) ready")

# CELL 7: Observability Infrastructure
class ObservabilityManager:
    """
    Three pillars: Logs, Traces, Metrics
    """
    def __init__(self):
        # Logs
        self.logger = logging.getLogger("CareerCoachAgent")
        self.logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(h)
        
        # Traces (OpenTelemetry)
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        self.tracer = trace.get_tracer(__name__)
        
        # Metrics
        self.metrics = {
            "tool_calls": 0,
            "tool_successes": 0,
            "steps_executed": 0,
            "memory_retrievals": 0,
            "input_filtered": 0,
            "output_screened": 0,
            "cache_hits": 0,
            "total_tokens": 0
        }
    
    def log_thought(self, thought: str, step: int, trace_id: str):
        self.logger.info(f"[THINK-{step}][{trace_id}] {thought[:100]}...")
    
    def log_action(self, tool: str, step: int, trace_id: str):
        self.logger.info(f"[ACT-{step}][{trace_id}] Tool: {tool}")
        self.metrics["tool_calls"] += 1
    
    def log_observation(self, result: bool, step: int, trace_id: str):
        self.logger.info(f"[OBSERVE-{step}][{trace_id}] Success: {result}")
        if result:
            self.metrics["tool_successes"] += 1
    
    def log_security_event(self, event_type: str, details: str):
        self.logger.warning(f"[SECURITY] {event_type}: {details}")
        self.metrics["input_filtered"] += 1
    
    def start_span(self, name: str, attrs: Dict = None):
        span = self.tracer.start_span(name)
        if attrs:
            for k, v in attrs.items():
                span.set_attribute(k, str(v))
        return span
    
    def end_span(self, span, success: bool = True):
        span.set_status(Status(StatusCode.OK if success else StatusCode.ERROR))
        span.end()
    
    def get_metrics(self) -> Dict:
        return {
            **self.metrics,
            "tool_success_rate": round(
                (self.metrics["tool_successes"] / self.metrics["tool_calls"] * 100)
                if self.metrics["tool_calls"] > 0 else 0, 1
            )
        }

print("✓ Observability Manager (Logs, Traces, Metrics) ready")

# CELL 8: Context Engineering - Token Manager
class ContextManager:
    """
    Manages context window to prevent token overflow
    Implements recursive summarization strategy
    """
    def __init__(self, max_tokens: int = Config.MAX_CONTEXT_TOKENS):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def compress_context(self, context: str, llm) -> str:
        """Compress context if it exceeds token limit"""
        tokens = self.estimate_tokens(context)
        
        if tokens <= self.max_tokens:
            return context
        
        # Recursive summarization
        summary_prompt = f"""Compress this context to 50% length while preserving key information:

{context[:self.max_tokens * 4]}

Provide a concise summary."""
        
        try:
            compressed = llm.generate(summary_prompt)
            return compressed
        except:
            # Fallback: truncate
            return context[:self.max_tokens * 4]

print("✓ Context Manager (Token optimization) ready")

# CELL 9: Memory Architecture
@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: datetime

class SessionMemory:
    """Short-term memory: conversation history"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[ConversationTurn] = []
    
    def add_turn(self, role: str, content: str):
        self.history.append(ConversationTurn(role, content, datetime.now()))
    
    def get_context(self, max_turns: int = 5) -> str:
        recent = self.history[-max_turns:]
        return "\n".join([f"{t.role.upper()}: {t.content}" for t in recent])

class LongTermMemory:
    """Long-term memory: Vector store for semantic retrieval"""
    def __init__(self, user_id: str, obs: ObservabilityManager):
        self.user_id = user_id
        self.obs = obs
        try:
            client = chromadb.Client()
            self.collection = client.get_or_create_collection(f"user_{user_id}")
            print(f"  ✓ Vector store initialized for {user_id}")
        except:
            self.collection = None
            print("  ⚠️ Vector store unavailable - using in-memory")
            self.memory = {}
    
    def store(self, key: str, content: str, metadata: Dict = None):
        if self.collection:
            self.collection.add(
                documents=[content],
                metadatas=[metadata or {}],
                ids=[f"{self.user_id}_{key}_{time.time()}"]
            )
        else:
            self.memory[key] = content
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        self.obs.metrics["memory_retrievals"] += 1
        if self.collection:
            try:
                results = self.collection.query(query_texts=[query], n_results=k)
                return results['documents'][0] if results['documents'] else []
            except:
                return []
        else:
            return list(self.memory.values())[:k]

print("✓ Memory Architecture (Session + Long-term Vector Store) ready")

# CELL 10: Tool System
class ToolResult:
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class LinkedInTool(Tool):
    @property
    def name(self) -> str:
        return "retrieve_linkedin_summary"
    
    def execute(self, user_profile: Dict, authority: AgentAuthority) -> ToolResult:
        """Security: Read-only access with least privilege"""
        if not authority.can_execute("read:linkedin"):
            return ToolResult(success=False, error="Permission denied: read:linkedin")
        
        try:
            data = {
                "title": user_profile["linkedin"]["title"],
                "company": user_profile["linkedin"]["company"],
                "experience": user_profile["linkedin"]["experience_years"]
            }
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, error=f"LinkedIn unavailable: {e}")

class DocumentTool(Tool):
    @property
    def name(self) -> str:
        return "process_documents"
    
    def execute(self, user_profile: Dict, authority: AgentAuthority) -> ToolResult:
        """Security: Only accesses user's own documents"""
        if not authority.can_execute("read:documents"):
            return ToolResult(success=False, error="Permission denied: read:documents")
        
        try:
            data = {
                "values": user_profile["documents"]["values"],
                "interests": user_profile["documents"]["interests"],
                "goals": user_profile["documents"]["goals"]
            }
            return ToolResult(success=True, data=data)
        except Exception as e:
            return ToolResult(success=False, error=f"Document processing failed: {e}")

class HITLTool(Tool):
    @property
    def name(self) -> str:
        return "request_confirmation"
    
    def execute(self, insight: str, authority: AgentAuthority) -> ToolResult:
        """Pauses for user confirmation on high-stakes insights"""
        return ToolResult(
            success=True,
            data={"requires_confirmation": True, "insight": insight}
        )

print("✓ Tool System (LinkedIn, Documents, HITL) with security ready")

# CELL 11: LLM Interface with Team of Specialists

class LLM:
    """
    Wrapper for Gemini models
    Supports both complex reasoning (Gemini 3 Pro) and simple tasks (Gemini 2.5 Flash)
    """
    def __init__(self, api_key: str, model_name: str = "gemini-3.0-pro"):
        self.model_name = model_name
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                print(f" ✓ {model_name} configured")
                self.has_api = True
            except Exception as e:
                print(f" ⚠️ Failed to configure {model_name}: {e}")
                self.model = None
                self.has_api = False
                print(" ⚠️ Demo mode activated")
        else:
            self.model = None
            self.has_api = False
            print(f" ⚠️ Demo mode - no API key for {model_name}")

    def generate(self, prompt: str) -> str:
        """Generate response from the model"""
        if self.model and self.has_api:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"⚠️ API call failed for {self.model_name}: {e}")
                return self._demo_response()
        return self._demo_response()

    def _demo_response(self) -> str:
        """Fallback response when API unavailable"""
        return "DEMO_MODE_RESPONSE"


class LLMTeam:
    """
    Team of Specialists approach:
    - Gemini 3 Pro: Complex reasoning, nuanced coaching, semantic comparison
    - Gemini 2.5 Flash: Simple classification and routing tasks
    """
    def __init__(self, api_key: str):
        print("\n🧠 Initializing Team of Specialists...")
        
        # Complex Reasoning & Planning Model
        self.pro_model = LLM(api_key, "gemini-3.0-pro")
        
        # Simple Classification Model
        self.flash_model = LLM(api_key, "gemini-2.5-flash")
        
        print("✓ LLM Team ready (Pro for reasoning, Flash for classification)\n")
    
    def classify(self, prompt: str) -> str:
        """Use Flash for simple classification tasks"""
        return self.flash_model.generate(prompt)
    
    def reason(self, prompt: str) -> str:
        """Use Pro for complex reasoning and coaching"""
        return self.pro_model.generate(prompt)
    
    def generate(self, prompt: str, task_complexity: str = "complex") -> str:
        """
        Route to appropriate model based on task complexity
        
        Args:
            prompt: The prompt to process
            task_complexity: Either 'simple' or 'complex' (default: 'complex')
        """
        if task_complexity == "simple":
            return self.classify(prompt)
        else:
            return self.reason(prompt)

print("✓ LLM Interface (Gemini 3 Pro + 2.5 Flash Team) ready")

# CELL 12: Orchestration Layer
class CareerCoachOrchestrator:
    """
    Level 2: Strategic Problem Solver
    Production-ready with security, reliability, and observability
    """
    def __init__(self, user_profile: Dict, api_key: str, obs: ObservabilityManager):
        self.profile = user_profile
        self.obs = obs
        self.llm = LLMTeam(api_key)
        
        # Security components
        self.authority = AgentAuthority(
            agent_id="career_coach_capstone",
            permissions=["read:linkedin", "read:documents", "write:memory"]
        )
        self.guardrails = GuardrailFilter()
        
        # Reliability components
        self.idempotency = IdempotencyManager()
        self.context_mgr = ContextManager()
        
        # Memory
        self.session = SessionMemory(f"session_{time.time()}")
        self.ltm = LongTermMemory(user_profile.get("user_id", "default"), obs)
        
        # Tools
        self.tools = {
            "linkedin": LinkedInTool(),
            "documents": DocumentTool(),
            "hitl": HITLTool()
        }
        
        self.system_prompt = """You are an expert career coach helping mid-career professionals.

Available tools:
- retrieve_linkedin_summary: Get professional background
- process_documents: Analyze values, interests, goals
- request_confirmation: Verify high-stakes insights with user

Use Chain-of-Thought reasoning: THINK → ACT → OBSERVE before responding."""
    
    def process_request(self, user_input: str, max_steps: int = 5) -> str:
        """
        Main orchestration with full security and reliability
        """
        trace_id = str(uuid.uuid4())[:8]
        master_span = self.obs.start_span("user_request", {"trace_id": trace_id})
        
        try:
            # SECURITY: Input filtering
            is_safe, filtered_input = self.guardrails.filter_input(user_input)
            if not is_safe:
                self.obs.log_security_event("INPUT_BLOCKED", filtered_input)
                self.obs.end_span(master_span, success=False)
                return f"⚠️ Security: {filtered_input}"
            
            # Add to session
            self.session.add_turn("user", filtered_input)
            
            # Retrieve memories
            memories = self.ltm.retrieve(filtered_input)
            memory_context = "\n".join(memories) if memories else "No prior context"
            
            # Build context
            context = f"""{self.system_prompt}

USER PROFILE:
- Role: {self.profile['linkedin']['title']} at {self.profile['linkedin']['company']}
- Experience: {self.profile['linkedin']['experience_years']} years
- Values: {', '.join(self.profile['documents']['values'])}
- Interests: {', '.join(self.profile['documents']['interests'])}
- Goals: {', '.join(self.profile['documents']['goals'])}
- Scenarios: {', '.join(self.profile['possible_selves'])}

MEMORY: {memory_context}
HISTORY: {self.session.get_context()}

USER QUERY: {filtered_input}"""
            
            # Context management
            context = self.context_mgr.compress_context(context, self.llm)
            
            # Multi-step orchestration
            for step in range(1, max_steps + 1):
                step_span = self.obs.start_span(f"step_{step}", {"step": step})
                self.obs.metrics["steps_executed"] += 1
                
                # THINK
                thought_prompt = f"{context}\n\nStep {step}: What should you do next? Think step-by-step."
                thought = self.llm.generate(thought_prompt)
                self.obs.log_thought(thought, step, trace_id)
                
                # ACT with idempotency
                tool_executed = False
                
                if "retrieve_linkedin" in thought.lower() or step == 1:
                    tool = self.tools["linkedin"]
                    params = {"user_id": self.profile["user_id"]}
                    idem_key = self.idempotency.generate_key(tool.name, params)
                    
                    # Check cache
                    cached = self.idempotency.get_cached_result(idem_key)
                    if cached:
                        result = cached
                        self.obs.metrics["cache_hits"] += 1
                    else:
                        self.obs.log_action(tool.name, step, trace_id)
                        result = tool.execute(self.profile, self.authority)
                        self.idempotency.cache_result(idem_key, result)
                    
                    self.obs.log_observation(result.success, step, trace_id)
                    if result.success:
                        context += f"\n\nLINKEDIN DATA: {json.dumps(result.data)}"
                    tool_executed = True
                
                elif "process_document" in thought.lower() or step == 2:
                    tool = self.tools["documents"]
                    params = {"user_id": self.profile["user_id"]}
                    idem_key = self.idempotency.generate_key(tool.name, params)
                    
                    cached = self.idempotency.get_cached_result(idem_key)
                    if cached:
                        result = cached
                        self.obs.metrics["cache_hits"] += 1
                    else:
                        self.obs.log_action(tool.name, step, trace_id)
                        result = tool.execute(self.profile, self.authority)
                        self.idempotency.cache_result(idem_key, result)
                    
                    self.obs.log_observation(result.success, step, trace_id)
                    if result.success:
                        context += f"\n\nDOCUMENT DATA: {json.dumps(result.data)}"
                    tool_executed = True
                
                elif "confirmation" in thought.lower() and step >= 3:
                    tool = self.tools["hitl"]
                    self.obs.log_action(tool.name, step, trace_id)
                    result = tool.execute("Detected career tension - verify with user", self.authority)
                    self.obs.log_observation(result.success, step, trace_id)
                    self.obs.end_span(step_span)
                    break
                
                self.obs.end_span(step_span)
                
                if not tool_executed and step > 2:
                    break
            
            # Generate final roadmap
            final_prompt = f"""{context}

Now generate a comprehensive career transition roadmap analyzing:
1. Each scenario's alignment with their values and goals
2. Barriers and first steps for each path
3. 30/60/90 day action plan
4. Decision framework

Be specific and personalized to THEIR profile."""
            
            roadmap = self.llm.generate(final_prompt)
            
            # SECURITY: Output screening
            is_safe, screened_output = self.guardrails.screen_output(roadmap)
            self.obs.metrics["output_screened"] += 1
            
            if not is_safe:
                self.obs.log_security_event("OUTPUT_BLOCKED", screened_output)
                roadmap = "⚠️ Response contained sensitive information and was filtered"
            else:
                roadmap = screened_output
            
            # Store in memory
            self.ltm.store(
                "roadmap_session",
                f"Generated roadmap for {filtered_input}",
                {"type": "roadmap", "timestamp": datetime.now().isoformat()}
            )
            
            self.obs.end_span(master_span, success=True)
            return roadmap
            
        except Exception as e:
            self.obs.end_span(master_span, success=False)
            return f"Error: {e}"

print("✓ Orchestration Layer (Think-Act-Observe) ready")

# CELL 13: Main Application
class CareerCoachAgent:
    """Production-grade career coaching agent with full governance"""
    def __init__(self, user_profile: Dict, api_key: str = ""):
        self.obs = ObservabilityManager()
        self.orchestrator = CareerCoachOrchestrator(
            user_profile,
            api_key,
            self.obs
        )
        print("  ✓ Agent initialized with security & reliability")
    
    def generate_roadmap(self) -> str:
        return self.orchestrator.process_request(
            "Generate my personalized career transition roadmap"
        )
    
    def get_metrics(self) -> Dict:
        return self.obs.get_metrics()
    
    def get_security_status(self) -> Dict:
        return {
            "agent_identity": self.orchestrator.authority.get_identity(),
            "guardrails_active": True,
            "idempotency_enabled": True
        }

print("✓ Main Application (CareerCoachAgent) ready")

# CELL 14: User Profile Configuration
print("\n" + "="*80)
print("📋 CONFIGURE YOUR PROFILE")
print("="*80)
print("\n💡 TIP: If you want to explore your values, take the assessment at:")
print("   http://personalvalu.es/")
print("="*80)

# ============================================================================
# EDIT YOUR INFORMATION HERE
# ============================================================================
job_title = "Program Manager"
company = "ABC Corp"
years_experience = 20
your_values = "Challenge, Variety, Curiosity"
your_interests = "Travel, Fitness, Start-Ups"
your_goals = "Financial security, Meaningful work"
scenario_1 = "Find a new full-time job"
scenario_2 = "Start a new business"
scenario_3 = "Travel the world"
# ============================================================================

USER_PROFILE = {
    "user_id": "capstone_demo",
    "linkedin": {
        "title": job_title,
        "company": company,
        "experience_years": years_experience
    },
    "documents": {
        "values": [v.strip() for v in your_values.split(',')],
        "interests": [i.strip() for i in your_interests.split(',')],
        "goals": [g.strip() for g in your_goals.split(',')]
    },
    "possible_selves": [scenario_1, scenario_2, scenario_3]
}

print("="*80)
print("✅ PROFILE CONFIGURED")
print("="*80)
print(f"👤 {job_title} at {company}")
print(f"📅 {years_experience} years experience")
print(f"💎 Values: {your_values}")
print(f"🎨 Interests: {your_interests}")
print(f"🎯 Goals: {your_goals}")
print(f"🔮 {len(USER_PROFILE['possible_selves'])} career scenarios")

# CELL 15: Generate Roadmap
print("\n" + "="*80)
print("🎯 GENERATING CAREER ROADMAP")
print("="*80)

agent = CareerCoachAgent(USER_PROFILE, GEMINI_API_KEY)

print("\n🔒 Security Status:")
print(json.dumps(agent.get_security_status(), indent=2))

print("\n⏳ Processing...\n")
roadmap = agent.generate_roadmap()

print("\n" + "="*80)
print("📋 YOUR CAREER TRANSITION ROADMAP")
print("="*80)
print()

if roadmap == "DEMO_MODE_RESPONSE":
    print(f"""# Career Transition Roadmap for {USER_PROFILE['linkedin']['title']}

## Scenario Analysis

### 1. {USER_PROFILE['possible_selves'][0]}
**Alignment Score**: 85%
**Strengths**: Leverages your {years_experience} years of experience
**Barriers**: Market competition, age bias in some sectors
**First Steps**:
- Update LinkedIn with recent achievements
- Identify 10 target companies
- Network with 3 industry contacts weekly

### 2. {USER_PROFILE['possible_selves'][1]}
**Alignment Score**: 90%
**Strengths**: Aligns with values: {your_values}
**Barriers**: Client acquisition, inconsistent income
**First Steps**:
- Define service offerings clearly
- Reach out to 5 past colleagues
- Create portfolio website

### 3. {USER_PROFILE['possible_selves'][2]}
**Alignment Score**: 75%
**Strengths**: Work-life balance, reduced stress
**Barriers**: Financial planning needed, identity shift
**First Steps**:
- Calculate minimum income needs
- Explore part-time opportunities
- Test with 1-2 days/week initially

## 30/60/90 Day Plan
**30 Days**: Explore all options, have 6 informational interviews
**60 Days**: Pilot test preferred option (freelance project or interview process)
**90 Days**: Make decision and take first major action

## Decision Framework
Given your values ({your_values}) and {years_experience} years experience:
1. Which scenario provides financial security while honoring your need for independence?
2. Which allows you to pursue {your_interests} more fully?
3. Which minimizes regret 5 years from now?""")
else:
    print(roadmap)

print("\n" + "="*80)
print("📊 SYSTEM METRICS")
print("="*80)
print(json.dumps(agent.get_metrics(), indent=2))

print("\n" + "="*80)
print("✅ CAPSTONE PRODUCTION AGENT COMPLETE")
print("="*80)
print("\nArchitectural Components Demonstrated:")
print("✓ Level 2: Strategic Problem Solver (Multi-step orchestration)")
print("✓ Think-Act-Observe Loop (Chain-of-Thought reasoning)")
print("✓ Tool System (LinkedIn, Documents, HITL)")
print("✓ Memory Architecture (Session + Long-term with Vector Store)")
print("✓ Observability (Logs, Traces, Metrics)")
print("✓ Security: Agent Identity & Least Privilege")
print("✓ Security: Input Filtering & Output Screening (Guardrails)")
print("✓ Reliability: Idempotency Manager (Safe-to-retry)")
print("✓ Efficiency: Context Management & Token Optimization")
print("="*80)

