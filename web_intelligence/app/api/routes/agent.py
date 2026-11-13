"""FastAPI routes for Stage 7 Agentic Research Orchestration."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from web_intelligence.app.agent.models import (
    AgentApprovalDecisionRequest,
    AgentGapsResponse,
    AgentPlanEditRequest,
    AgentPlanResponse,
    AgentResearchRequest,
    AgentSessionSummary,
    AgentTasksResponse,
    AgentTraceResponse,
    ResearchSession,
    ToolDescriptor,
)
from web_intelligence.app.agent.orchestrator import AgentSessionNotFoundError
from web_intelligence.app.agent.task_manager import PlanValidationError
from web_intelligence.app.dependencies import AppContainer
from web_intelligence.app.llm.security import redact_llm_error

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def _get_agent(request: Request):
    container: AppContainer = request.app.state.container
    if container.agent_orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agentic research orchestrator is not initialized.",
        )
    return container, container.agent_orchestrator


def _raise_http(exc: Exception) -> None:
    detail = redact_llm_error(exc)
    if isinstance(exc, AgentSessionNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from exc
    if isinstance(exc, (PlanValidationError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        ) from exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error executing agent research operation.",
    ) from exc


@router.get("/tools", response_model=list[ToolDescriptor])
async def list_registered_tools(request: Request) -> list[ToolDescriptor]:
    """List controlled registered tools exposed by ToolRegistry."""
    _, agent = _get_agent(request)
    return agent.list_tools()


@router.post(
    "/research",
    response_model=ResearchSession,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_agent_research(
    payload: AgentResearchRequest,
    request: Request,
) -> ResearchSession:
    """Launch an autonomous multi-iteration research session asynchronously."""
    _, agent = _get_agent(request)
    try:
        return await agent.start_research(payload, background=True)
    except Exception as exc:
        _raise_http(exc)


@router.get("/research", response_model=list[AgentSessionSummary])
async def list_agent_sessions(
    request: Request,
    limit: int = Query(default=30, ge=1, le=100),
) -> list[AgentSessionSummary]:
    """List persisted autonomous research sessions newest first."""
    container, _ = _get_agent(request)
    return await container.repository.list_agent_sessions(limit=limit)


@router.get("/research/{research_id}", response_model=ResearchSession)
async def get_agent_session(research_id: str, request: Request) -> ResearchSession:
    """Inspect full persisted state, metrics, evaluation, and synthesis for a session."""
    container, _ = _get_agent(request)
    session = await container.repository.get_agent_session(research_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent research session '{research_id}' not found.",
        )
    return session


@router.get("/research/{research_id}/plan", response_model=AgentPlanResponse)
async def get_agent_plan(research_id: str, request: Request) -> AgentPlanResponse:
    """Return the current versioned ResearchPlanSpec DAG for inspection."""
    container, _ = _get_agent(request)
    session = await container.repository.get_agent_session(research_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent research session '{research_id}' not found.",
        )
    return AgentPlanResponse(
        research_id=session.research_id,
        status=session.status,
        plan=session.research_plan,
    )


@router.put("/research/{research_id}/plan", response_model=AgentPlanResponse)
async def update_agent_plan(
    research_id: str,
    payload: AgentPlanEditRequest,
    request: Request,
) -> AgentPlanResponse:
    """Inspect and update pending plan tasks cleanly without disturbing completed tasks."""
    container, agent = _get_agent(request)
    try:
        updated_plan = await agent.update_plan(research_id, payload)
        session = await container.repository.get_agent_session(research_id)
        return AgentPlanResponse(
            research_id=research_id,
            status=session.status if session else updated_plan and "pending",
            plan=updated_plan,
        )
    except Exception as exc:
        _raise_http(exc)


@router.get("/research/{research_id}/tasks", response_model=AgentTasksResponse)
async def get_agent_tasks(research_id: str, request: Request) -> AgentTasksResponse:
    """Inspect completed, pending, and failed research task queues."""
    container, _ = _get_agent(request)
    session = await container.repository.get_agent_session(research_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent research session '{research_id}' not found.",
        )
    return AgentTasksResponse(
        research_id=session.research_id,
        status=session.status,
        current_task_id=session.current_task_id,
        completed_tasks=session.completed_tasks,
        pending_tasks=session.pending_tasks,
        failed_tasks=session.failed_tasks,
    )


@router.get("/research/{research_id}/trace", response_model=AgentTraceResponse)
async def get_agent_trace(
    research_id: str,
    request: Request,
    limit: int = Query(default=300, ge=1, le=1000),
) -> AgentTraceResponse:
    """Inspect chronological operational trace entries explaining every agent decision."""
    container, _ = _get_agent(request)
    session = await container.repository.get_agent_session(research_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent research session '{research_id}' not found.",
        )
    entries = await container.repository.list_agent_trace_entries(research_id, limit=limit)
    return AgentTraceResponse(
        research_id=session.research_id,
        status=session.status,
        entries=entries,
    )


@router.get("/research/{research_id}/gaps", response_model=AgentGapsResponse)
async def get_agent_gaps(research_id: str, request: Request) -> AgentGapsResponse:
    """Inspect identified research gaps, evaluation report, and stopping decision."""
    container, _ = _get_agent(request)
    session = await container.repository.get_agent_session(research_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent research session '{research_id}' not found.",
        )
    unresolved = sum(1 for g in session.identified_gaps if not g.resolved)
    resolved = sum(1 for g in session.identified_gaps if g.resolved)
    return AgentGapsResponse(
        research_id=session.research_id,
        status=session.status,
        gaps=session.identified_gaps,
        unresolved_count=unresolved,
        resolved_count=resolved,
        evidence_evaluation=session.evidence_evaluation,
        stopping_decision=session.stopping_decision,
    )


@router.post("/research/{research_id}/pause", response_model=ResearchSession)
async def pause_agent_research(research_id: str, request: Request) -> ResearchSession:
    """Request clean checkpoint pause for an active agent session."""
    _, agent = _get_agent(request)
    try:
        return await agent.pause_session(research_id)
    except Exception as exc:
        _raise_http(exc)


@router.post("/research/{research_id}/resume", response_model=ResearchSession)
async def resume_agent_research(research_id: str, request: Request) -> ResearchSession:
    """Resume a paused or interrupted session asynchronously without repeating completed tasks."""
    _, agent = _get_agent(request)
    try:
        return await agent.resume_session(research_id, background=True)
    except Exception as exc:
        _raise_http(exc)


@router.post("/research/{research_id}/cancel", response_model=ResearchSession)
async def cancel_agent_research(research_id: str, request: Request) -> ResearchSession:
    """Cancel an active or paused session cleanly."""
    _, agent = _get_agent(request)
    try:
        return await agent.cancel_session(research_id)
    except Exception as exc:
        _raise_http(exc)


@router.post("/research/{research_id}/approve", response_model=ResearchSession)
async def approve_agent_action(
    research_id: str,
    payload: AgentApprovalDecisionRequest,
    request: Request,
) -> ResearchSession:
    """Approve or reject a pending external tool execution request."""
    _, agent = _get_agent(request)
    try:
        return await agent.handle_approval(research_id, payload, background=True)
    except Exception as exc:
        _raise_http(exc)
