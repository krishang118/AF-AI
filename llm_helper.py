"""
LLM Helper for Assumption-Driven Forecasting
Supports multiple providers: Ollama (local) and Groq (API)
"""

import json
import re
import requests
from typing import Dict, List, Optional, Any
from forecast_engine import Assumption, Event, AssumptionType, EventType

# Model configuration
MODELS = {
    "ollama": {
        "llama3:latest": {"display": "Llama 3 (local)", "fast": True},
        "deepseek-r1:7b": {"display": "DeepSeek-R1 7B (local)", "reasoning": True}
    },
    "groq": {
        "openai/gpt-oss-20b": {"display": "openai/gpt-oss-20B (Groq)", "fast": True}
    },
    "openai": {
        "gpt-4o-mini": {"display": "GPT-4o Mini (OpenAI)", "fast": True},
        "gpt-4o": {"display": "GPT-4o (OpenAI)", "advanced": True}
    }
}


class LLMHelper:
    """Interface to LLM providers (Ollama local, Groq API, or OpenAI API)"""
    
    def __init__(self, provider: str = "ollama", model: str = "llama3:latest", 
                 api_key: Optional[str] = None, base_url: str = "http://localhost:11434"):
        """
        Initialize LLM helper with specified provider and model
        
        Args:
            provider: "ollama", "groq", or "openai"
            model: Model name (llama3:latest, deepseek-r1:7b, openai/gpt-oss-20b, gpt-4o-mini, or gpt-4o)
            api_key: API key (required for groq or openai provider)
            base_url: Ollama base URL (default: http://localhost:11434)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Initialize client based on provider
        if provider == "groq":
            if not api_key:
                raise ValueError("Groq API key required for groq provider")
            try:
                from groq import Groq
                import httpx
                # Use custom httpx client to fix SSL issues on Windows
                http_client = httpx.Client(verify=False)
                self.groq_client = Groq(api_key=api_key, http_client=http_client)
            except ImportError as e:
                if "groq" in str(e):
                    raise ImportError("Groq SDK not installed. Run: pip install groq")
                raise ImportError("httpx not installed. Run: pip install httpx")
        else:
            self.groq_client = None
        
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required for openai provider")
            try:
                from openai import OpenAI
                import httpx
                # Use custom httpx client to fix SSL issues on Windows
                http_client = httpx.Client(verify=False)
                self.openai_client = OpenAI(api_key=api_key, http_client=http_client)
            except ImportError as e:
                if "openai" in str(e):
                    raise ImportError("OpenAI SDK not installed. Run: pip install openai")
                raise ImportError("httpx not installed. Run: pip install httpx")
        else:
            self.openai_client = None
    
    def get_model_display_name(self) -> str:
        """Get human-readable model name"""
        if self.provider in MODELS and self.model in MODELS[self.provider]:
            return MODELS[self.provider][self.model]["display"]
        return f"{self.model} ({self.provider})"
    
    def _call_groq(self, prompt: str, system: str = "") -> str:
        """Make API call to Groq"""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return json.dumps({"error": f"Groq API Error: {str(e)}"})
    
    def _call_openai(self, prompt: str, system: str = "") -> str:
        """Make API call to OpenAI"""
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return json.dumps({"error": f"OpenAI API Error: {str(e)}"})
        
    def _call_ollama(self, prompt: str, system: str = "", force_json: bool = False) -> str:
        """Make API call to Ollama with optional JSON validation"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more deterministic outputs
                    "top_p": 0.9
                }
            }
            
            if force_json:
                payload["format"] = "json"
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # Clean DeepSeek <think> tags and markdown wrappers
            if '<think>' in response_text:
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # Clean markdown code blocks
            if '```' in response_text:
                response_text = re.sub(r'^```json\s*', '', response_text)
                response_text = re.sub(r'^```\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            # Validate JSON if required
            if force_json:
                try:
                    json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from substring if mixed content
                    try:
                        match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                        if match:
                            response_text = match.group(1)
                            json.loads(response_text)
                        else:
                            raise
                    except:
                        return json.dumps({"error": "Invalid JSON response", "raw": response_text})
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"LLM Error: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Unexpected error: {str(e)}"})
    
    def _call(self, prompt: str, system: str = "", force_json: bool = False) -> str:
        """Unified call method that routes to appropriate provider"""
        if self.provider == "groq":
            # Groq doesn't support force_json directly, rely on prompt engineering
            return self._call_groq(prompt, system)
        elif self.provider == "openai":
            # OpenAI doesn't support force_json directly, rely on prompt engineering
            return self._call_openai(prompt, system)
        else:  # ollama
            return self._call_ollama(prompt, system, force_json)

    
    def interpret_column_name(self, column_name: str, sample_values: List[Any]) -> Dict[str, str]:
        """Interpret ambiguous column names"""
        system = "You are a data analyst helping interpret column names in Excel/CSV files. Be concise."
        
        prompt = f"""
Column name: "{column_name}"
Sample values: {sample_values[:5]}

What metric does this column likely represent? 
Respond in JSON format:
{{
  "metric_type": "revenue|users|volume|price|other",
  "standard_name": "suggested column name",
  "confidence": "high|medium|low"
}}
"""
        
        response = self._call(prompt, system)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "metric_type": "other",
                "standard_name": column_name,
                "confidence": "low"
            }
    
    def suggest_assumptions_from_data(self, metric: str, historical_data: List[float], 
                                     periods: List[str]) -> List[Dict[str, Any]]:
        """Suggest plausible assumptions based on historical patterns"""
        system = """You are a data analyst helping identify trends in time series data.
CRITICAL RULES:
- ALWAYS provide suggestions if the data has 3 or more points.
- Do NOT reject data with 5-10 points; this is sufficient for trend analysis.
- Only reject data if there are FEWER than 3 points.
- If assumptions are weak, flag uncertainty but still provide them.
- Be concise - max 2 sentences per reasoning
- Only suggest if you have confidence"""
        
        prompt = f"""
Metric: {metric}
Historical values: {historical_data}
Periods: {periods}

Analyze this data and suggest 2-3 reasonable assumptions for forecasting.
Consider: growth trends, volatility, structural breaks.

IMPORTANT: If the data is too limited or volatile to make reliable suggestions, 
respond with: {{"insufficient_data": true, "reason": "explanation"}}

Otherwise respond in this JSON format:
[
  {{
    "type": "growth",
    "name": "descriptive name",
    "value": numeric_value,
    "confidence": "high|medium|low",
    "reasoning": "1-2 sentence explanation"
  }}
]
"""
        
        response = self._call(prompt, system, force_json=True)
        
        try:
            result = json.loads(response)
            
            # Check for insufficient data flag
            if isinstance(result, dict):
                if result.get('insufficient_data'):
                    return []
                # Handle wrapped responses (e.g. {"suggestions": [...]})
                if 'suggestions' in result and isinstance(result['suggestions'], list):
                    result = result['suggestions']
                elif 'assumptions' in result and isinstance(result['assumptions'], list):
                    result = result['assumptions']
            
            # Validate structure
            if isinstance(result, list):
                validated = []
                for item in result:
                    if all(k in item for k in ['type', 'name', 'value', 'confidence']):
                        validated.append(item)
                return validated
            return []
        except json.JSONDecodeError:
            return []
    
    def parse_analyst_notes(self, text: str) -> List[Dict[str, Any]]:
        """Parse unstructured analyst notes into structured assumptions"""
        system = "You extract structured assumptions and events from analyst notes."
        
        prompt = f"""
Analyst notes:
{text}

Extract any assumptions, forecasts, or planned events mentioned in the text.
Respond in JSON format:

{{
  "assumptions": [
    {{
      "type": "growth|pricing|market_share|other",
      "metric": "affected metric",
      "value": numeric_value_if_mentioned,
      "notes": "original text snippet"
    }}
  ],
  "events": [
    {{
      "type": "product_launch|price_change|regulation|other",
      "name": "event name",
      "date": "YYYY-MM-DD or YYYY or null",
      "impact": "description of expected impact"
    }}
  ]
}}
"""
        
        response = self._call(prompt, system)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"assumptions": [], "events": []}
    
    def explain_forecast_drivers(self, metric: str, assumptions: List[Assumption], 
                                events: List[Event], forecast_values: List[float], 
                                period_count: int = None) -> str:
        """Generate explanation of what's driving the forecast"""
        system = """You are a precise financial forecasting analyst. Your role is to:
1. Accurately interpret mathematical impacts (multipliers <1.0 = reductions, >1.0 = increases)
2. Assess event severity based on BOTH magnitude AND decay duration (in periods)
3. Explain forecast drivers based ONLY on provided assumptions and events
4. Use consistent terminology (reduction/decline for negatives, increase/growth for positives)
5. Never hallucinate business drivers not in the data
6. Assess impact duration using decay PERIODS, not time units (weeks/months/years)
7. Use comparative language: "fewer periods" vs "more periods" for decay assessment"""
        
        assumptions_text = "\n".join([
            f"- {a.name}: {a.value:.1%} ({a.type.value})"
            for a in assumptions if a.metric == metric
        ])
        
        # Enhanced event formatting with explicit direction logic
        events_list = []
        for e in events:
            if e.metric == metric:
                multiplier = e.impact_multiplier
                pct_change = (multiplier - 1) * 100
                
                # Explicit direction logic
                if multiplier > 1.0:
                    direction = "POSITIVE (increases metric)"
                    impact_desc = f"+{pct_change:.1f}% boost"
                elif multiplier < 1.0:
                    direction = "NEGATIVE (decreases metric)"
                    impact_desc = f"{pct_change:.1f}% reduction"
                else:
                    direction = "NEUTRAL"
                    impact_desc = "no change"
                
                # Decay severity assessment (period-agnostic)
                if e.decay_periods == 0:
                    decay_note = ", **permanent effect** (no decay)"
                    severity = "HIGH"
                elif e.decay_periods <= 2:
                    decay_note = f", **decays over {e.decay_periods} periods**"
                    severity = "LOW"
                elif e.decay_periods <= 5:
                    decay_note = f", **decays over {e.decay_periods} periods**"
                    severity = "MODERATE"
                elif e.decay_periods <= 12:
                    decay_note = f", **decays over {e.decay_periods} periods**"
                    severity = "SIGNIFICANT"
                else:
                    decay_note = f", **decays over {e.decay_periods} periods**"
                    severity = "HIGH"
                
                events_list.append(
                    f"- **{e.name}** on {e.date}: {impact_desc} ({multiplier:.2f}× multiplier) - {direction}{decay_note} [Severity: {severity}]"
                )
        
        events_text = "\n".join(events_list) if events_list else "None"
        
        prompt = f"""
TASK: Explain what drives the {metric} forecast.

FORECAST VALUES: {forecast_values}

GROWTH ASSUMPTIONS:
{assumptions_text or "None"}

DISCRETE EVENTS:
{events_text}

CRITICAL INSTRUCTIONS:
1. Start with baseline growth trajectory from assumptions
2. Identify any events and their impact direction (positive increase vs negative reduction)
3. **ASSESS EVENT SEVERITY**: Consider BOTH magnitude AND decay duration
   - Events with fewer decay periods (1-5) recover faster, less sustained impact
   - Events with more decay periods (10+) have prolonged impact, slower recovery
   - DO NOT assume periods are weeks, months, or years - use relative terms only
4. Describe the combined effect on the forecast curve
5. Use 3-4 concise sentences
6. DO NOT invent business drivers (AOV, churn, etc.) unless explicitly in assumptions
7. Be mathematically precise and grounded in the provided data ONLY
8. When discussing decay, use "N periods" not "N weeks/months/years"

OUTPUT FORMAT:
- Baseline: [describe growth assumptions]
- Events: [describe each event's impact direction, magnitude, AND decay duration in periods with severity]
- Net Effect: [overall forecast trajectory considering decay patterns]
"""
        
        return self._call(prompt, system)
    
    def answer_what_if(self, question: str, context: Dict[str, Any], conversation_history: list = None) -> str:
        """Answer scenario and 'what if' questions with conversation context awareness"""
        system = """You are an expert financial analyst assistant. 
Your role is to analyze structured data, forecasts, and business scenarios with mathematical precision.

CRITICAL RULES:
1. Event multipliers <1.0 are NEGATIVE (reductions/declines)
2. Event multipliers >1.0 are POSITIVE (increases/growth)
3. Base assumptions ONLY on provided context - never hallucinate drivers
4. Use precise mathematical language and consistent terminology
5. Assess event severity based on BOTH magnitude AND decay duration (in periods)
6. Use period-relative language: "fewer decay periods" vs "more decay periods"
7. DO NOT assume periods are weeks, months, or years - stay period-agnostic"""
        
        context_text = json.dumps(context, indent=2, default=str)
# Quick manual fix - insert this block right after line 379 in llm_helper.py

        # Format recent conversation history for context
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Get last 4 exchanges (8 messages max) for context
            recent_history = conversation_history[-8:]
            history_lines = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:200]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            
            if history_lines:
                history_context = f"""
RECENT CONVERSATION CONTEXT (for disambiguating references):
{chr(10).join(history_lines)}

IMPORTANT: Use this conversation history to understand what the user's current question refers to.
For example:
- If user previously asked for a "rating out of 10" and now mentions "8", they likely mean the rating scale (8/10), not 8%.
- If user asks about improving a "4/10 score to 8", they want to improve the rating, NOT achieve 8% growth.
- Pay attention to whether user is discussing growth rates (%) vs qualitative ratings (X/10).
"""
        
        prompt = f"""
Question: {question}

{history_context}
ANALYSIS RULES:
1. Events with multiplier < 1.0 are NEGATIVE (reductions/declines)
2. Events with multiplier > 1.0 are POSITIVE (increases/growth)
3. Base assumptions on ONLY the provided context, do not hallucinate
4. Use precise mathematical language
5. **DECAY AWARENESS**: Fewer decay periods = faster recovery. More decay periods = prolonged impact.
6. **NO TEMPORAL ASSUMPTIONS**: Use "N periods" not "N weeks/months/years"

CONTEXT:
{context_text}

DATA VISIBILITY CONFIRMATION:
- You have access to 'uploaded_sheets_summary' (raw files as uploaded by user)
- You have access to 'combined_data_summary' (merged dataset used for forecasting)
- You have access to 'assumptions' (all growth/pricing assumptions)
- You have access to 'events' (discrete events with multipliers and decay_periods)
- You have access to 'scenarios' (forecast values: base, upside, downside)

RESPONSE INSTRUCTIONS:
- If greeting: respond briefly
- If asking about data: cite specific numbers from context
- If asking about drivers: reference actual assumptions/events by name AND decay duration in PERIODS
- If event has multiplier 0.5, describe as "50% reduction" NOT "50% boost"
- If event has multiplier 1.5, describe as "50% increase" or "boost"
- When discussing events, describe decay as "decays over N periods" not "over N weeks/months"
- Compare decay periods relatively: "fewer periods" vs "more periods"
- Be mathematically accurate and consistent
- Provide insights in 2-4 concise bullets or sentences
- GROUND EVERYTHING in the provided data - do not invent metrics like AOV, churn rate, customer lifetime value unless they are explicitly mentioned in the context

ADDITIONAL GUIDANCE:
- Analyze 'combined_data_summary' (if present) for the final merged dataset
- Check 'uploaded_sheets_summary' for details on individual source files  
- Use 'assumptions' and 'scenarios' to explain forecast drivers
- If CAGR or growth rates are mentioned, be precise about calculations
- If discussing event impact, mention both the multiplier AND the decay period count for full context
"""
        
        return self._call(prompt, system)
    
    def summarize_for_slides(self, metric: str, scenarios: Dict[str, List[float]], 
                            assumptions: List[Assumption]) -> str:
        """Generate executive summary for presentation slides"""
        system = "You write concise executive summaries for strategy presentations."
        
        base_growth = ((scenarios['base'][-1] / scenarios['base'][0]) - 1) * 100
        upside_growth = ((scenarios['upside'][-1] / scenarios['upside'][0]) - 1) * 100
        downside_growth = ((scenarios['downside'][-1] / scenarios['downside'][0]) - 1) * 100
        
        prompt = f"""
Create EXACTLY 3 bullets for {metric} forecast (no intro, no conclusion):

Base: {base_growth:.1f}% | Upside: {upside_growth:.1f}% | Downside: {downside_growth:.1f}%

Top assumptions: {chr(10).join([f"- {a.name}" for a in assumptions[:3]])}

Format:
- [Most important driver]
- [Key assumption/risk]
- [Uncertainty or range]

Be ruthlessly concise. No filler words.
"""
        
        return self._call(prompt, system)
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_available = any(m.get('name', '').startswith(self.model.split(':')[0]) for m in models)
            
            return {
                "status": "connected",
                "model_available": model_available,
                "available_models": [m.get('name') for m in models]
            }
            
        except requests.exceptions.RequestException:
            return {
                "status": "disconnected",
                "model_available": False,
                "error": "Ollama not running or not accessible"
            }


class AssumptionBuilder:
    """Helper to convert LLM outputs into Assumption objects"""
    
    @staticmethod
    def from_llm_suggestion(suggestion: Dict[str, Any], metric: str) -> Optional[Assumption]:
        """Convert LLM suggestion to Assumption object"""
        try:
            type_map = {
                'growth': AssumptionType.GROWTH,
                'event': AssumptionType.EVENT,
                'pricing': AssumptionType.PRICING,
                'market_share': AssumptionType.MARKET_SHARE,
            }
            
            assumption_type = type_map.get(suggestion.get('type', 'growth'), AssumptionType.GROWTH)
            
            return Assumption(
                id=f"llm_{metric}_{len(str(suggestion))}",
                type=assumption_type,
                name=suggestion.get('name', 'LLM Suggestion'),
                metric=metric,
                value=float(suggestion.get('value', 0)),
                confidence=suggestion.get('confidence', 'medium'),
                source='llm',
                notes=suggestion.get('reasoning', '')
            )
        except (ValueError, KeyError):
            return None
    
    @staticmethod
    def from_analyst_note(note_data: Dict[str, Any]) -> Optional[Assumption]:
        """Convert parsed analyst note to Assumption"""
        try:
            type_map = {
                'growth': AssumptionType.GROWTH,
                'pricing': AssumptionType.PRICING,
                'market_share': AssumptionType.MARKET_SHARE,
            }
            
            assumption_type = type_map.get(note_data.get('type', 'growth'), AssumptionType.GROWTH)
            
            return Assumption(
                id=f"note_{len(str(note_data))}",
                type=assumption_type,
                name=note_data.get('notes', 'From analyst notes')[:50],
                metric=note_data.get('metric', 'unknown'),
                value=float(note_data.get('value', 0)),
                confidence='medium',
                source='analyst',
                notes=note_data.get('notes', '')
            )
        except (ValueError, KeyError):
            return None


class EventBuilder:
    """Helper to convert LLM outputs into Event objects"""
    
    @staticmethod
    def from_llm_event(event_data: Dict[str, Any], metric: str) -> Optional[Event]:
        """Convert LLM-parsed event to Event object"""
        try:
            type_map = {
                'product_launch': EventType.PRODUCT_LAUNCH,
                'price_change': EventType.PRICE_CHANGE,
                'regulation': EventType.REGULATION,
                'market_entry': EventType.MARKET_ENTRY,
                'market_exit': EventType.MARKET_EXIT,
            }
            
            event_type = type_map.get(event_data.get('type', 'product_launch'), EventType.PRODUCT_LAUNCH)
            
            return Event(
                id=f"event_{metric}_{len(str(event_data))}",
                event_type=event_type,
                name=event_data.get('name', 'Unnamed Event'),
                metric=metric,
                date=event_data.get('date', '2025-01-01'),
                impact_multiplier=1.0,  # Default, user can adjust
                notes=event_data.get('impact', '')
            )
        except (ValueError, KeyError):
            return None
