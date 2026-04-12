from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew
import os

app = FastAPI(title="Digital Godfather Backend")

# CORS — Vercel se connect hone ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Groq API Key — Railway Environment Variable se
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── AGENTS DEFINE ──
def create_agents():
    vision_agent = Agent(
        role="Vision & Pitch Expert",
        goal="Create compelling pitch decks and investor materials",
        backstory="Top startup pitch consultant with 100+ successful fundraises",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False
    )

    market_agent = Agent(
        role="Market Research Analyst",
        goal="Analyze markets, competitors, and opportunities",
        backstory="Ex-McKinsey analyst specializing in Indian startup ecosystem",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False
    )

    revenue_agent = Agent(
        role="CFO & Financial Expert",
        goal="Build financial models and pricing strategies",
        backstory="Startup CFO who helped 50+ companies achieve profitability",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False
    )

    brand_agent = Agent(
        role="Brand & Marketing Strategist",
        goal="Build brand identity and marketing campaigns",
        backstory="CMO who built 3 unicorn brands from scratch",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False
    )

    return {
        "vision": vision_agent,
        "market": market_agent,
        "revenue": revenue_agent,
        "brand": brand_agent
    }

# ── REQUEST MODELS ──
class AgentRequest(BaseModel):
    startup_name: str
    startup_idea: str
    agent_type: str  # vision, market, revenue, brand

class ReportRequest(BaseModel):
    startup_name: str
    startup_idea: str
    report_type: str  # overview, problem, market, etc.

# ── HEALTH CHECK ──
@app.get("/")
def root():
    return {
        "status": "Digital Godfather is LIVE",
        "agents": ["vision", "market", "revenue", "brand"],
        "powered_by": "CrewAI + Groq"
    }

# ── RUN AGENT ──
@app.post("/run-agent")
async def run_agent(request: AgentRequest):
    try:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        agents = create_agents()

        if request.agent_type not in agents:
            raise HTTPException(status_code=400, detail=f"Invalid agent: {request.agent_type}")

        agent = agents[request.agent_type]

        # Task prompts per agent
        task_prompts = {
            "vision": f"""
                Create a complete pitch package for startup "{request.startup_name}": {request.startup_idea}
                
                Deliver:
                1. ELEVATOR PITCH (60 words max)
                2. PITCH DECK OUTLINE (10 slides with key points)
                3. INVESTOR ONE-PAGER (Problem, Solution, Market, Ask)
                
                India market focus. Real numbers. No fluff.
            """,
            "market": f"""
                Complete market analysis for "{request.startup_name}": {request.startup_idea}
                
                Deliver:
                1. COMPETITOR TABLE (5 real competitors, funding, weakness)
                2. MARKET SIZE (TAM/SAM/SOM in INR for India)
                3. OPPORTUNITY (Why now, key insights, timing)
                
                Use real 2026 data. India focus.
            """,
            "revenue": f"""
                Financial strategy for "{request.startup_name}": {request.startup_idea}
                
                Deliver:
                1. PRICING TIERS (3 tiers with INR prices)
                2. 12-MONTH MODEL (Monthly revenue, costs, net)
                3. UNIT ECONOMICS (CAC, LTV, payback period)
                
                Realistic India market numbers.
            """,
            "brand": f"""
                Brand strategy for "{request.startup_name}": {request.startup_idea}
                
                Deliver:
                1. BRAND VOICE (5 words + do's and don'ts)
                2. MESSAGING FRAMEWORK (Tagline, value prop, 3 messages)
                3. 30-DAY CONTENT CALENDAR (Instagram + LinkedIn daily posts)
                
                India audience. Engaging tone.
            """
        }

        task = Task(
            description=task_prompts[request.agent_type],
            expected_output="Detailed, actionable startup intelligence report",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=False
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "agent": request.agent_type,
            "startup": request.startup_name,
            "result": str(result)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── RUN ALL AGENTS ──
@app.post("/run-all-agents")
async def run_all_agents(request: AgentRequest):
    try:
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        results = {}

        for agent_type in ["vision", "market", "revenue", "brand"]:
            single_request = AgentRequest(
                startup_name=request.startup_name,
                startup_idea=request.startup_idea,
                agent_type=agent_type
            )
            response = await run_agent(single_request)
            results[agent_type] = response["result"]

        return {
            "status": "success",
            "startup": request.startup_name,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
