"""
╔══════════════════════════════════════════════════════════════╗
║         DIGITAL GODFATHER — RAILWAY BACKEND                  ║
║         FastAPI + CrewAI + Groq (llama-3.3-70b)             ║
║         SSE Streaming · CORS · Health Check                  ║
╚══════════════════════════════════════════════════════════════╝

ENV VARIABLES required on Railway:
  GROQ_API_KEY   = your Groq API key from console.groq.com
  PORT           = (Railway sets this automatically)
"""

import os
import json
import asyncio
import traceback
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ── Load .env (local dev only — Railway uses env vars directly) ──
load_dotenv()

# ══════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════
app = FastAPI(
    title="Digital Godfather Core",
    description="Autonomous AI Startup Factory Backend",
    version="3.0.0"
)

# CORS — allow ALL origins so frontend connects instantly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════
# GROQ LLM SETUP
# Using langchain-groq (most stable on Railway)
# Falls back to litellm if langchain-groq fails
# ══════════════════════════════════════════
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

def get_llm():
    """
    Returns a LangChain-compatible LLM.
    Tries langchain-groq first, falls back to litellm wrapper.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set on Railway.")

    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.7,
            max_tokens=4096,
        )
    except ImportError:
        # litellm fallback
        try:
            from litellm import LiteLLM
            return LiteLLM(
                model=f"groq/{GROQ_MODEL}",
                api_key=GROQ_API_KEY,
                temperature=0.7,
            )
        except Exception as e:
            raise RuntimeError(f"LLM init failed: {e}")

# ══════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════
class DeployRequest(BaseModel):
    startup_name: str
    startup_idea: str

class AgentRequest(BaseModel):
    startup_name: str
    startup_idea: str
    agent_type:   str
    agent_role:   Optional[str] = ""
    context:      Optional[str] = ""
    selected_agents: Optional[list] = []

class RunRequest(BaseModel):
    startup_name: str
    startup_idea: str

# ══════════════════════════════════════════
# AGENT LIBRARY (38 agents — mirrors frontend)
# ══════════════════════════════════════════
AGENT_LIBRARY = {
    # Strategy
    "ceo":           {"name": "CEO Agent",              "cat": "Strategy",    "role": "Chief Executive — sets vision, final decisions"},
    "cto":           {"name": "CTO Agent",              "cat": "Strategy",    "role": "Chief Technology Officer — tech architecture"},
    "cfo":           {"name": "CFO Agent",              "cat": "Strategy",    "role": "Chief Financial Officer — financial models"},
    "cmo":           {"name": "CMO Agent",              "cat": "Strategy",    "role": "Chief Marketing Officer — brand strategy"},
    "strategy":      {"name": "Strategy Agent",         "cat": "Strategy",    "role": "Strategy Consultant — competitive positioning"},
    # Research
    "market_research":{"name": "Market Research Agent", "cat": "Research",   "role": "Market Analyst — TAM/SAM/SOM analysis"},
    "competitor":    {"name": "Competitor Intel Agent", "cat": "Research",    "role": "Competitive Intelligence — deep competitor teardown"},
    "customer":      {"name": "Customer Research Agent","cat": "Research",    "role": "Customer Insights — personas, JTBD"},
    "trend":         {"name": "Trend Analyst Agent",    "cat": "Research",    "role": "Trend Forecaster — 2026-2030 market signals"},
    "data":          {"name": "Data Science Agent",     "cat": "Research",    "role": "Data Scientist — analytics, KPI dashboards"},
    # Legal
    "legal":         {"name": "Legal Shark Agent",      "cat": "Legal",       "role": "Startup Legal Counsel — contracts, term sheets"},
    "gst":           {"name": "GST Expert Agent",       "cat": "Legal",       "role": "GST & Tax Specialist — India tax optimization"},
    "compliance":    {"name": "Compliance Agent",       "cat": "Legal",       "role": "Compliance Officer — RBI, SEBI, MCA regulations"},
    "ip":            {"name": "IP Agent",               "cat": "Legal",       "role": "IP Specialist — patents, trademarks"},
    "contract":      {"name": "Contract Agent",         "cat": "Legal",       "role": "Contract Drafter — NDAs, vendor agreements"},
    # Product & Tech
    "product":       {"name": "Product Manager Agent",  "cat": "Product",     "role": "Product Manager — roadmap, OKRs"},
    "uiux":          {"name": "UI/UX Designer Agent",   "cat": "Product",     "role": "UI/UX Designer — wireframes, design system"},
    "app_dev":       {"name": "App Developer Agent",    "cat": "Product",     "role": "Mobile Developer — React Native / Flutter"},
    "web_dev":       {"name": "Web Developer Agent",    "cat": "Product",     "role": "Full Stack Developer — system architecture"},
    "ai_ml":         {"name": "AI/ML Agent",            "cat": "Product",     "role": "AI Engineer — ML pipeline, LLM integration"},
    "security":      {"name": "Security Agent",         "cat": "Product",     "role": "Security Engineer — VAPT, threat modeling"},
    "devops":        {"name": "DevOps Agent",           "cat": "Product",     "role": "DevOps Engineer — CI/CD, cloud infra"},
    # Operations
    "ops":           {"name": "Operations Agent",       "cat": "Operations",  "role": "Operations Manager — SOPs, efficiency"},
    "supply_chain":  {"name": "Logistics Expert Agent", "cat": "Operations",  "role": "Supply Chain — vendor, last-mile India"},
    "hr":            {"name": "HR Agent",               "cat": "Operations",  "role": "HR Manager — hiring, ESOP, culture"},
    "finance_ops":   {"name": "Finance Ops Agent",      "cat": "Operations",  "role": "Finance Manager — P&L, runway, cash flow"},
    "paperclip":     {"name": "Paperclip Agent",        "cat": "Operations",  "role": "Infinite Optimizer — eliminates ALL waste, loops forever"},
    # Sales & Marketing
    "sales":         {"name": "Sales Agent",            "cat": "Sales",       "role": "Sales Manager — funnel, pitch scripts"},
    "growth":        {"name": "Growth Hacker Agent",    "cat": "Sales",       "role": "Growth Manager — viral loops, AARRR"},
    "content":       {"name": "Content Agent",          "cat": "Sales",       "role": "Content Strategist — SEO, thought leadership"},
    "seo_agent":     {"name": "SEO Agent",              "cat": "Sales",       "role": "SEO Specialist — keywords, backlinks"},
    "ads":           {"name": "Paid Ads Agent",         "cat": "Sales",       "role": "Performance Marketer — Google/Meta Ads India"},
    "community":     {"name": "Community Agent",        "cat": "Sales",       "role": "Community Manager — WhatsApp, Discord, Reddit"},
    # Finance & Funding
    "investor_rel":  {"name": "Investor Relations Agent","cat": "Finance",    "role": "Investor Relations — data room, warm intros"},
    "pitch":         {"name": "Pitch Agent",            "cat": "Finance",     "role": "Pitch Coach — deck, story, objections"},
    "valuation":     {"name": "Valuation Agent",        "cat": "Finance",     "role": "Valuation Expert — DCF, VC method India"},
    "grant":         {"name": "Grant Agent",            "cat": "Finance",     "role": "Grant Specialist — DPIIT, Startup India"},
    "fundraise":     {"name": "Fundraise Agent",        "cat": "Finance",     "role": "Fundraising Strategist — SAFE, term sheets"},
    # Brand & Partnerships
    "brand":         {"name": "Brand Agent",            "cat": "Brand",       "role": "Brand Strategist — identity, voice, positioning"},
    "pr":            {"name": "PR Agent",               "cat": "Brand",       "role": "PR Specialist — press, YourStory, Inc42"},
    "partner":       {"name": "Partnership Agent",      "cat": "Brand",       "role": "Business Development — strategic partnerships"},
}

# ══════════════════════════════════════════
# SMART AGENT SELECTOR
# Uses LLM to pick top 5 agents for a venture
# ══════════════════════════════════════════
async def select_agents_for_venture(startup_name: str, startup_idea: str) -> list[str]:
    """Ask LLM to select the 5 most relevant agents for this specific startup."""
    agent_list = "\n".join([
        f"- {aid}: {info['role']}"
        for aid, info in AGENT_LIBRARY.items()
    ])

    prompt = f"""You are the Manager Agent for a trillion-dollar AI startup factory.

Startup: "{startup_name}" — {startup_idea}

Available agents:
{agent_list}

Pick EXACTLY 5 agent IDs most critical for this startup's first 90 days.
Consider: what type of startup is this? What does it need MOST right now?

Reply ONLY with a JSON array of agent IDs.
Example: ["ceo","market_research","product","legal","growth"]
No explanation. Only the JSON array."""

    try:
        llm = get_llm()
        # Use invoke for langchain-groq ChatGroq
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
        else:
            text = str(llm(prompt))

        # Parse the JSON array from response
        import re
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            ids_raw = match.group(0)
            ids = json.loads(ids_raw)
            # Validate each id exists in AGENT_LIBRARY
            valid = [i for i in ids if i in AGENT_LIBRARY]
            if len(valid) >= 3:
                return valid[:5]
    except Exception as e:
        print(f"[Manager] Agent selection failed: {e}")

    # Smart keyword fallback
    idea_lower = (startup_name + " " + startup_idea).lower()
    defaults = ["ceo"]
    if any(w in idea_lower for w in ["app", "software", "platform", "saas", "tech"]):
        defaults += ["product", "web_dev", "growth"]
    elif any(w in idea_lower for w in ["logistics", "delivery", "supply", "warehouse"]):
        defaults += ["supply_chain", "ops", "finance_ops"]
    elif any(w in idea_lower for w in ["legal", "compliance", "contract"]):
        defaults += ["legal", "compliance", "market_research"]
    elif any(w in idea_lower for w in ["brand", "media", "content", "social"]):
        defaults += ["brand", "content", "pr"]
    else:
        defaults += ["market_research", "product", "growth"]

    if "fundrais" in idea_lower or "investor" in idea_lower:
        defaults.append("pitch")
    elif "legal" not in defaults:
        defaults.append("legal")

    return list(dict.fromkeys(defaults))[:5]  # dedupe + limit


# ══════════════════════════════════════════
# SSE HELPER
# ══════════════════════════════════════════
def sse_line(data: dict) -> str:
    """Format a single SSE event line."""
    return f"data: {json.dumps(data)}\n\n"

def sse_log(message: str, level: str = "info") -> str:
    return sse_line({"type": "log", "level": level, "message": message})

def sse_agent(agent_id: str, agent_name: str, status: str = "hired") -> str:
    return sse_line({"type": "agent", "id": agent_id, "name": agent_name, "status": status})

def sse_result(text: str, agents: list) -> str:
    return sse_line({"type": "result", "text": text, "agents": agents})

def sse_error(message: str) -> str:
    return sse_line({"type": "error", "message": message})

def sse_done() -> str:
    return sse_line({"type": "done"})


# ══════════════════════════════════════════
# CREWAI AGENT RUNNER
# ══════════════════════════════════════════
async def run_crewai_agent(
    agent_id: str,
    startup_name: str,
    startup_idea: str,
    context: str = "",
    queue: asyncio.Queue = None
) -> str:
    """
    Runs a single CrewAI agent task for the given startup.
    Streams log lines to queue if provided.
    Returns the agent's text output.
    """
    try:
        from crewai import Agent, Task, Crew, Process

        agent_info = AGENT_LIBRARY.get(agent_id, {
            "name": agent_id,
            "role": "Startup Specialist",
            "cat": "General"
        })

        if queue:
            await queue.put(sse_log(f"// {agent_info['name']}: Starting analysis...", "info"))

        llm = get_llm()

        # Build the CrewAI agent
        agent = Agent(
            role=agent_info["role"],
            goal=f"Provide deep, actionable {agent_info['cat']} analysis for startup '{startup_name}'",
            backstory=f"""You are {agent_info['name']}, a world-class specialist with 15+ years 
            building successful startups in India. You give specific, data-driven advice with 
            real company names, INR numbers, and India-specific insights.""",
            llm=llm,
            verbose=False,
            allow_delegation=False,
        )

        ctx_text = f"\n\nContext from other agents:\n{context}" if context else ""

        task = Task(
            description=f"""Analyze startup "{startup_name}": {startup_idea}
            
Your role: {agent_info['role']}

Provide a comprehensive analysis including:
1. Expert analysis specific to your domain (3-4 concrete insights)
2. Immediate action items for next 30 days (5 specific tasks with owners)
3. Key risks in your domain and mitigation strategies
4. India-specific recommendations with real data points
5. One bold contrarian insight most people miss

Format with ### headers and ** bold ** for key points.
Be specific — use real company names, real numbers, real India market data.{ctx_text}""",
            agent=agent,
            expected_output="Detailed startup analysis report with specific insights, action items, risks, and India recommendations."
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        if queue:
            await queue.put(sse_log(f"// {agent_info['name']}: Running CrewAI task...", "info"))

        # Run in thread pool to avoid blocking event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            crew.kickoff
        )

        output = str(result)
        if hasattr(result, 'raw'):
            output = result.raw
        elif hasattr(result, 'output'):
            output = result.output

        if queue:
            await queue.put(sse_log(f"// ✓ {agent_info['name']}: Complete", "success"))

        return output

    except Exception as e:
        err_msg = f"// ✗ {agent_id}: {str(e)}"
        if queue:
            await queue.put(sse_log(err_msg, "error"))
        print(f"[CrewAI Error] {agent_id}: {traceback.format_exc()}")
        # Return a graceful fallback instead of crashing
        return f"### {AGENT_LIBRARY.get(agent_id, {}).get('name', agent_id)} Report\n\n*Note: AI processing encountered an issue. Please retry.*\n\nError: {str(e)}"


# ══════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "Digital Godfather Core",
        "version": "3.0.0",
        "status": "online",
        "agents": len(AGENT_LIBRARY),
        "groq_configured": bool(GROQ_API_KEY)
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Frontend polls this on load to check Railway connectivity.
    """
    return {
        "status": "ok",
        "service": "Digital Godfather Core",
        "groq_configured": bool(GROQ_API_KEY),
        "agents_available": len(AGENT_LIBRARY)
    }


@app.post("/deploy")
async def deploy_venture(req: DeployRequest):
    """
    Main deploy endpoint with SSE streaming.
    
    Flow:
    1. Manager Agent selects 5 best agents → streams each hire
    2. Each agent runs CrewAI task → streams logs + result
    3. Returns final JSON with all outputs
    
    Frontend receives a stream of SSE events:
      {type: "log",    level: "info|success|warn|error", message: "..."}
      {type: "agent",  id: "ceo", name: "CEO Agent", status: "hired"}
      {type: "result", text: "...", agents: [...]}
      {type: "error",  message: "..."}
      {type: "done"}
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()

        async def produce():
            try:
                # Step 1: Manager selects team
                await queue.put(sse_log("// Manager Agent: Scanning 38 agents...", "info"))
                await queue.put(sse_log(f"// Analyzing: {req.startup_name}", "info"))
                await asyncio.sleep(0.3)

                selected_ids = await select_agents_for_venture(req.startup_name, req.startup_idea)
                await queue.put(sse_log(f"// Manager selected: {', '.join(selected_ids)}", "success"))

                # Step 2: Announce hires one by one
                for agent_id in selected_ids:
                    info = AGENT_LIBRARY.get(agent_id, {"name": agent_id})
                    await asyncio.sleep(0.4)
                    await queue.put(sse_agent(agent_id, info["name"], "hired"))
                    await queue.put(sse_log(f"// ✓ Hired: {info['name']}", "success"))

                await asyncio.sleep(0.5)
                await queue.put(sse_log("// Team assembled. Beginning analysis...", "info"))

                # Step 3: Run each agent
                context = ""
                all_results = {}

                for agent_id in selected_ids:
                    info = AGENT_LIBRARY.get(agent_id, {"name": agent_id})
                    await queue.put(sse_log(f"// {info['name']}: Working...", "stream"))

                    result_text = await run_crewai_agent(
                        agent_id=agent_id,
                        startup_name=req.startup_name,
                        startup_idea=req.startup_idea,
                        context=context[:1000] if context else "",
                        queue=queue
                    )

                    all_results[agent_id] = result_text
                    # Build context for next agent
                    context += f"\n\n--- {info['name']} ---\n{result_text[:400]}"

                    await queue.put(sse_agent(agent_id, info["name"], "done"))

                # Step 4: Send final result
                combined = "\n\n".join([
                    f"## {AGENT_LIBRARY.get(k, {}).get('name', k)}\n\n{v}"
                    for k, v in all_results.items()
                ])

                await queue.put(sse_result(combined, selected_ids))
                await queue.put(sse_log("// 🎉 All agents complete!", "success"))

            except Exception as e:
                await queue.put(sse_error(f"SYSTEM ERROR: {str(e)}"))
                print(f"[/deploy Error]: {traceback.format_exc()}")
            finally:
                await queue.put(None)  # Sentinel to close stream

        # Start producer in background
        asyncio.create_task(produce())

        # Consume queue and yield SSE events
        while True:
            item = await queue.get()
            if item is None:
                yield sse_done()
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Important for Railway nginx
        }
    )


@app.post("/run-agent")
async def run_single_agent(req: AgentRequest):
    """
    Run a single specialist agent with SSE streaming.
    Called when user taps a specific agent card.
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()

        async def produce():
            try:
                info = AGENT_LIBRARY.get(req.agent_type, {
                    "name": req.agent_type,
                    "role": req.agent_role or "Startup Specialist",
                    "cat": "General"
                })

                await queue.put(sse_log(f"// {info['name']}: Activated", "info"))
                await queue.put(sse_log(f"// Startup: {req.startup_name}", ""))

                result_text = await run_crewai_agent(
                    agent_id=req.agent_type,
                    startup_name=req.startup_name,
                    startup_idea=req.startup_idea,
                    context=req.context or "",
                    queue=queue
                )

                await queue.put(sse_result(result_text, [req.agent_type]))
                await queue.put(sse_log(f"// ✓ {info['name']} complete", "success"))

            except Exception as e:
                await queue.put(sse_error(str(e)))
                print(f"[/run-agent Error]: {traceback.format_exc()}")
            finally:
                await queue.put(None)

        asyncio.create_task(produce())

        while True:
            item = await queue.get()
            if item is None:
                yield sse_done()
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/run")
async def run_sync(req: RunRequest):
    """
    Synchronous /run endpoint (non-streaming).
    Returns JSON directly — simpler alternative if SSE has issues.
    """
    try:
        selected_ids = await select_agents_for_venture(req.startup_name, req.startup_idea)

        results = {}
        for agent_id in selected_ids[:3]:  # Limit to 3 for sync speed
            try:
                result = await run_crewai_agent(
                    agent_id=agent_id,
                    startup_name=req.startup_name,
                    startup_idea=req.startup_idea,
                    context="",
                    queue=None
                )
                results[agent_id] = result
            except Exception as e:
                results[agent_id] = f"Error: {str(e)}"

        return JSONResponse({
            "status": "success",
            "startup_name": req.startup_name,
            "agents": selected_ids,
            "plan": f"AI team assembled for {req.startup_name}: {', '.join(selected_ids)}",
            "steps": [
                f"Step 1: {AGENT_LIBRARY.get(selected_ids[0], {}).get('name', selected_ids[0])} leads strategy",
                f"Step 2: Team analysis across {len(selected_ids)} domains",
                "Step 3: Reports ready in Command Center",
                "Step 4: Deploy & execute"
            ],
            "results": results
        })

    except Exception as e:
        print(f"[/run Error]: {traceback.format_exc()}")
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500
        )


@app.get("/agents")
async def list_agents():
    """Returns the full agent library — useful for frontend validation."""
    return {
        "total": len(AGENT_LIBRARY),
        "agents": [
            {"id": k, "name": v["name"], "cat": v["cat"], "role": v["role"]}
            for k, v in AGENT_LIBRARY.items()
        ]
    }


# ══════════════════════════════════════════
# ENTRYPOINT — Railway compatible
# ══════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"

    print(f"""
╔══════════════════════════════════════════╗
║   Digital Godfather Core — Starting      ║
║   Host: {host}  Port: {port}            ║
║   GROQ_API_KEY: {'SET ✓' if GROQ_API_KEY else 'MISSING ✗'}            ║
║   Agents loaded: {len(AGENT_LIBRARY)}                        ║
╚══════════════════════════════════════════╝
    """)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        timeout_keep_alive=120,
    )
