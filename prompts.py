from typing import Optional

import rich
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt


class AnswerStatement(BaseModel):
    reasoning: str = Field(
        description="Reasoning behind the answer. This should explain how the answer was derived."
    )
    answer_sentece: str = Field(
        description="Sentence of the answer. May be a title or paragraph text in markdown format. Should be a freeform text."
    )
    reference_url: Optional[str] = Field(
        description="URL of the reference from given context"
    )
    reference_title: Optional[str] = Field(
        description="Title of the reference from given context"
    )


class AnswerWithCitations(BaseModel):
    statements: list[AnswerStatement] = Field(
        description="Sequential list of sentences of the answer. Some statements are with references."
    )

    def format_response(self):
        answer_text = []
        for statement in self.statements:
            text = statement.answer_sentece
            if statement.reference_url and statement.reference_title:
                text += f" [[{statement.reference_title}]({statement.reference_url})]"
            answer_text.append(text)
        return "\n".join(answer_text)


class AnswersCollection(BaseModel):
    answers: list[AnswerWithCitations] = Field(
        description="List of answers with citations."
    )


class QueryInput(BaseModel):
    question: str = Field(
        description="Question to be answered based on multiple elements of the context."
    )
    context: list[Document] = Field(
        description="Context to be used for answering the question. If not provided, the retriever will be used to get the context."
    )


class SummaryInput(BaseModel):
    messages: list = Field(description="List of messages to summarize.")


class SummaryPrompt(PydanticPrompt[SummaryInput, AnswerWithCitations]):
    instruction = """Given a list of messages, generate a concise summary that captures the main points and key information.
    The summary should be clear, coherent, and well-structured.
    The summary should cover the research question and not the technical aspects of getting the context.
    """
    input_model = SummaryInput
    output_model = AnswerWithCitations
    examples = [
        (
            SummaryInput(
                messages=[
                    {
                        "content": "Genoa Cricket and Football Club is the oldest extant football team in Italy. They play at the Stadio Luigi Ferraris.",
                        "url": "https://en.wikipedia.org/wiki/Genoa_CFC",
                        "title": "Genoa CFC",
                    },
                    {
                        "content": "Stadio Luigi Ferraris is named after Luigi Ferraris, an Italian footballer and engineer.",
                        "url": "https://en.wikipedia.org/wiki/Stadio_Luigi_Ferraris",
                        "title": "Stadio Luigi Ferraris",
                    },
                    {
                        "content": "Luigi Ferraris was born in 1887.",
                        "url": "https://en.wikipedia.org/wiki/Luigi_Ferraris",
                        "title": "Luigi Ferraris",
                    },
                ],
            ),
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="Question asks for the emperor of China, answer is entitled with his name.",
                        answer_sentece="## Guangxu Emperor",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The context provides a summary of the main topic.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The oldest extant football team in Italy is Genoa CFC, which plays at the Stadio Luigi Ferraris.",
                        answer_sentece="* The oldest extant football team in Italy is Genoa CFC, which plays at the Stadio Luigi Ferraris.",
                        reference_url="https://en.wikipedia.org/wiki/Genoa_CFC",
                        reference_title="Genoa CFC",
                    ),
                    AnswerStatement(
                        reasoning="Question asks about age of Luigi Ferraris, who is the namesake of the stadium.",
                        answer_sentece="* The stadium is named after Luigi Ferraris, who was born in 1887.",
                        reference_url="https://en.wikipedia.org/wiki/Luigi_Ferraris",
                        reference_title="Luigi Ferraris",
                    ),
                    AnswerStatement(
                        reasoning="As Luigi Ferraris was born in 1887 and was 5 years old in 1892, the emperor of China at that time was the Guangxu Emperor.",
                        answer_sentece="* When Luigi Ferraris was 5 years old, in 1892, the emperor of China was the Guangxu Emperor.",
                        reference_url="https://en.wikipedia.org/wiki/Guangxu_Emperor",
                        reference_title="Guangxu Emperor",
                    ),
                ]
            ),
        ),
    ]


class FinalReportPrompt(PydanticPrompt[AnswerWithCitations, AnswerWithCitations]):
    instruction = """Given research findings and insights, generate a final report that summarizes the key points.
    The report should be clear, coherent, and well-structured, with a focus on the most important information without missing details.
    Do not report technical information about the research process.
    """
    input_model = AnswersCollection
    output_model = AnswerWithCitations
    examples = [
        (
            # Input: research findings assembled earlier
            AnswersCollection(
                answers=[
                    AnswerWithCitations(
                        statements=[
                            AnswerStatement(
                                reasoning="Provide a concise, informative title for the final report.",
                                answer_sentece="## Who ruled China in 1892 (derived from context)",
                                reference_url=None,
                                reference_title=None,
                            ),
                            AnswerStatement(
                                reasoning="Set reader expectations by stating that the answer is synthesized from context.",
                                answer_sentece="Here is the what context provides to your question:",
                                reference_url=None,
                                reference_title=None,
                            ),
                            AnswerStatement(
                                reasoning="Identify the oldest extant Italian team and its home stadium.",
                                answer_sentece="* The oldest extant football team in Italy is Genoa CFC, which plays at the Stadio Luigi Ferraris.",
                                reference_url="https://en.wikipedia.org/wiki/Genoa_CFC",
                                reference_title="Genoa CFC",
                            ),
                            AnswerStatement(
                                reasoning="Connect the stadium name to Luigi Ferraris and note his birth year.",
                                answer_sentece="* The stadium is named after Luigi Ferraris, who was born in 1887.",
                                reference_url="https://en.wikipedia.org/wiki/Luigi_Ferraris",
                                reference_title="Luigi Ferraris",
                            ),
                            AnswerStatement(
                                reasoning="Infer the year when Ferraris was 5 (1892) and identify the emperor.",
                                answer_sentece="* When Luigi Ferraris was 5 years old, in 1892, the emperor of China was the Guangxu Emperor.",
                                reference_url="https://en.wikipedia.org/wiki/Guangxu_Emperor",
                                reference_title="Guangxu Emperor",
                            ),
                        ]
                    )
                ]
            ),
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="State the final answer prominently as the report heading.",
                        answer_sentece="## Final Answer: Guangxu Emperor (China, 1892)",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Explain that the conclusion is derived from multiple contextual facts.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Support: identify Genoa CFC and its stadium.",
                        answer_sentece="* Genoa CFC is Italy's oldest extant football team and plays at Stadio Luigi Ferraris.",
                        reference_url="https://en.wikipedia.org/wiki/Genoa_CFC",
                        reference_title="Genoa CFC",
                    ),
                    AnswerStatement(
                        reasoning="Support: establish Luigi Ferraris' birth year.",
                        answer_sentece="* The stadium's namesake, Luigi Ferraris, was born in 1887.",
                        reference_url="https://en.wikipedia.org/wiki/Luigi_Ferraris",
                        reference_title="Luigi Ferraris",
                    ),
                    AnswerStatement(
                        reasoning="Support: identify the emperor of China in 1892, given the inferred year.",
                        answer_sentece="* Therefore, in 1892 (when Ferraris was 5), the emperor of China was the Guangxu Emperor.",
                        reference_url="https://en.wikipedia.org/wiki/Guangxu_Emperor",
                        reference_title="Guangxu Emperor",
                    ),
                ]
            ),
        ),
        (
            # Input: findings about Kubernetes origins and governance
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="Provide a neutral title summarizing the topic.",
                        answer_sentece="## Kubernetes: Origins and Governance",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Orient the reader that the bullets reflect gathered findings.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Define Kubernetes and its origin at Google.",
                        answer_sentece="* Kubernetes is an open-source container orchestration system originally designed by Google.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Provide the initial public release year (2014).",
                        answer_sentece="* Initial public release was in 2014.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Explain CNCF stewardship beginning in 2015.",
                        answer_sentece="* Google donated Kubernetes to the Cloud Native Computing Foundation (CNCF) in 2015.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Summarize key platform capabilities with a canonical source.",
                        answer_sentece="* Key capabilities include container scheduling, scaling, and rolling updates.",
                        reference_url="https://kubernetes.io/docs/concepts/overview/",
                        reference_title="Kubernetes Docs: Overview",
                    ),
                ]
            ),
            # Output: final report that synthesizes and sequences the points clearly
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="Provide a clear, outcome-focused title for the final report.",
                        answer_sentece="## Kubernetes: Origins, Release Timeline, and Governance",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Set up the synthesized summary section.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="Summarize definition and provenance.",
                        answer_sentece="* Kubernetes is an open-source container orchestration system that originated at Google.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Note the initial release timing.",
                        answer_sentece="* The project was initially released to the public in 2014.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Indicate transition to CNCF for governance.",
                        answer_sentece="* In 2015, Google donated Kubernetes to the CNCF, which now stewards the project.",
                        reference_url="https://en.wikipedia.org/wiki/Kubernetes",
                        reference_title="Kubernetes",
                    ),
                    AnswerStatement(
                        reasoning="Summarize core operational features with vendor-neutral docs.",
                        answer_sentece="* Core capabilities include container scheduling, autoscaling, and rolling updates for safer deployments.",
                        reference_url="https://kubernetes.io/docs/concepts/overview/",
                        reference_title="Kubernetes Docs: Overview",
                    ),
                ]
            ),
        ),
    ]


class ClarificationOutput(BaseModel):
    clarification_is_required: bool = Field(
        description="Whether a clarifying question is required to better understand the user's query."
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="Clarifying question to be asked to the user. In case clarification_is_required is False, this field should be empty.",
    )
    final_statements: list[str] = Field(
        description="List of statements from the user inputs to conduct search on"
    )


class ClarificationInput(BaseModel):
    messages: list[str] = Field(description="List of messages from the user so far.")


class ClarificationPrompt(PydanticPrompt[ClarificationInput, ClarificationOutput]):
    instruction = """Decide whether a clarifying question is required to better understand the user's query.
    If a clarifying question is required, generate the question to be asked to the user.
    If no clarifying question is required, provide the final statements to conduct search on.
    The final statements should be a list of strings that represent the user's input.
    Ask for clarification only if the user's query is ambiguous or lacks sufficient detail.
    Do it only twice in a row max.
    Provide no more than 3 final statements.
    """
    input_model = ClarificationInput
    output_model = ClarificationOutput
    examples = [
        (
            ClarificationInput(
                messages=[
                    "What is the capital of France?",
                ]
            ),
            ClarificationOutput(
                clarification_is_required=False,
                clarifying_question=None,
                final_statements=[
                    "Capital of France",
                ],
            ),
        ),
        (
            ClarificationInput(
                messages=[
                    "Tell me about the Python programming language.",
                    "I want to know about its history and features.",
                ]
            ),
            ClarificationOutput(
                clarification_is_required=False,
                clarifying_question=None,
                final_statements=[
                    "Python programming language history" "Python language features",
                ],
            ),
        ),
        (
            ClarificationInput(
                messages=[
                    "What are the health benefits of eating apples?",
                ],
            ),
            ClarificationOutput(
                clarification_is_required=True,
                clarifying_question="Are you interested in the nutritional benefits, or the benefits for specific health conditions?",
                final_statements=[],
            ),
        ),
        (
            ClarificationInput(
                messages=[
                    "Can you tell me about the book '1984'?",
                    "I want to know about its author and main themes.",
                ]
            ),
            ClarificationOutput(
                clarification_is_required=False,
                clarifying_question=None,
                final_statements=[
                    "Book '1984' author",
                    "Book '1984' main themes",
                ],
            ),
        ),
    ]


class PlannerOutput(BaseModel):
    reasoning_on_topics: str = Field(description="Reasoning statements for each topic.")
    expanded_topics: list[str] = Field(
        description="Expanded and refined list of distinct topics to research, no more than 10 topics."
    )
    reasoning_on_plan: str = Field(description="Reasoning statements for the plan. Plan should have exactly 3 steps.")
    plan: list[str] = Field(
        description="List of steps in the plan. Only 3 steps are allowed. These should be 3 very distinct research topics that cover the expended topics. Each step should be concise description of research activity."
    )


class PlannerInput(BaseModel):
    messages: str = Field(description="Recent messages from the conversation")
    search_topics: list[str] = Field(description="List of initial topics to research")


class PlanPrompt(PydanticPrompt[PlannerInput, PlannerOutput]):
    instruction = """You are an expert planner. Your task is to create a research plan based on the user's request.
    You will be provided with a list of topics to research. Your task is to expand and refine this list into a comprehensive set of topics that cover the user's request.
    You will also create a plan consisting of strictly 3 distinct very specific steps, each focusing on a different aspect of the research topics.
    Each step should be a concise description of the research to be conducted. 1-3 sentences are allowed for each step.
    Each step should answer just a single very specific question.
    Make sure that the steps are distinct and cover different aspects of the research topics.
    """

    input_model = PlannerInput
    output_model = PlannerOutput
    examples = [
        (
            # Example 1: Consumer decision (gravel bike)
            PlannerInput(
                messages=(
                    "User goal: Find a new gravel bike for long mixed-terrain rides (fire roads, light singletrack). "
                    "Budget around $2,500. Priorities: comfort over speed, tubeless tires, endurance geometry. "
                    "Rider height 5'10\" (178 cm). No brand preference."
                ),
                search_topics=[
                    "Research new bikes",
                    "Find a new gravel bike",
                ],
            ),
            PlannerOutput(
                reasoning_on_topics=(
                    "Break the generic request into concrete, decision-making topics that directly affect suitability: "
                    "intended terrain and riding profile, budget-constrained model shortlist (2024–2025), fit and geometry for 5'10\" rider, "
                    "frame material trade-offs, drivetrain configuration (1x vs 2x) for mixed terrain, tire clearance and wheel/tire compatibility (tubeless), "
                    "braking and mounting points for endurance comfort, and authoritative review sources for triangulation. Limit to <=10 distinct topics."
                ),
                expanded_topics=[
                    "Riding profile and terrain requirements (endurance, mixed surface)",
                    "Model shortlist 2024–2025 within ~$2,500 budget",
                    "Fit guidance and endurance geometry for 5'10\" rider",
                    "Frame material trade-offs (alloy vs carbon vs steel) for comfort",
                    "Drivetrain choice trade-offs (1x vs 2x) for varied gradients",
                    "Tire clearance, tubeless compatibility, and wheel standards",
                    "Mounting points (bags, extra bottles) and comfort features",
                    "Reliable review sources and long-ride test impressions",
                ],
                reasoning_on_plan=(
                    "Choose exactly three non-overlapping steps that each answer a single, specific question while covering the decision funnel: "
                    "(1) confirm a budget-constrained model pool; (2) determine fit/geometry for the rider; (3) compare key build trade-offs for endurance comfort."
                ),
                plan=[
                    "Which 2024–2025 gravel bike models under $2,500 are best-reviewed for endurance comfort on mixed terrain (list 5–8 models with sources)?",
                    "What size and endurance-oriented geometry characteristics best fit a 5'10\" (178 cm) rider among those models (reach, stack, size guidance)?",
                    "For long mixed-terrain rides at this budget, what are the practical trade-offs between frame materials and 1x vs 2x drivetrains for comfort and range?",
                ],
            ),
        ),
        (
            # Example 2: Fact-finding (current monarch)
            PlannerInput(
                messages=(
                    "User asks: Who is the King of England? Provide authoritative, current information with context (regnal name, accession and coronation dates, "
                    "and a note on the proper title referring to the United Kingdom)."
                ),
                search_topics=[
                    "Current UK monarch",
                    "Who is the King of England",
                ],
            ),
            PlannerOutput(
                reasoning_on_topics=(
                    "Expand to discrete, checkable facts from authoritative sources: identity and regnal name; accession and coronation timeline; "
                    "and official title terminology distinguishing 'King of England' vs 'King of the United Kingdom'."
                ),
                expanded_topics=[
                    "Identity of the current monarch of the United Kingdom",
                    "Regnal name and styling",
                    "Accession date and circumstances",
                    "Coronation date and key ceremony details",
                    "Official title/terminology guidance (UK vs 'England')",
                    "Authoritative sources (Royal Family, UK government, BBC, Britannica)",
                ],
                reasoning_on_plan=(
                    "Sequence the research to ensure verification before interpretation: first confirm identity and regnal details; "
                    "then verify key dates; finally clarify official styling. Each step is a single, narrowly defined question."
                ),
                plan=[
                    "Who is the current monarch of the United Kingdom and what is the correct regnal name?",
                    "What are the accession and coronation dates for the current monarch (with 2+ authoritative citations)?",
                    "What is the proper official title and why is 'King of England' imprecise in modern usage?",
                ],
            ),
        ),
        (
            # Example 3: Technology history/governance (Kubernetes)
            PlannerInput(
                messages=(
                    "User request: Summarize Kubernetes' origin and governance. Include where it started, first public release timing, "
                    "and how CNCF stewards the project today. Keep steps distinct and precise."
                ),
                search_topics=[
                    "Kubernetes origin",
                    "Kubernetes CNCF governance",
                ],
            ),
            PlannerOutput(
                reasoning_on_topics=(
                    "Split into origin/provenance, public release timing, and transition to CNCF governance; also capture high-level capabilities to provide context "
                    "but keep the plan focused on verifiable historical/governance facts."
                ),
                expanded_topics=[
                    "Kubernetes origin at Google (internal history, project roots)",
                    "Initial public release timing (year/version)",
                    "Donation to CNCF (year and rationale)",
                    "Current CNCF governance model (SIGs, TOC, maintainers)",
                    "Official sources (kubernetes.io, CNCF blog, Wikipedia)",
                ],
                reasoning_on_plan=(
                    "Address the user's facets in three narrow questions: origin, CNCF transition, and current governance structure."
                ),
                plan=[
                    "Where did Kubernetes originate and in what year was it first publicly released?",
                    "When was Kubernetes donated to the Cloud Native Computing Foundation and why?",
                    "How is Kubernetes governed today under CNCF (key bodies and roles)?",
                ],
            ),
        ),
    ]


class CitationPrompt(PydanticPrompt[QueryInput, AnswerWithCitations]):
    instruction = """Generate answer to the question based on the given context.
    Answer may consist of multiple sentences, some of them may be with references.
    Come up with a full answer step-by-step, using multiple elements of the context.
    Answer should be in markdown format following the template:
    ```
    # Title
    Here is the what context provides to your question:
    * statement 1
    * statement 2 [reference title](reference url)
    ```
    """
    input_model = QueryInput
    output_model = AnswerWithCitations
    examples = [
        (
            QueryInput(
                question="What is the main topic of the document?",
                context=[
                    Document(
                        page_content="The main topic is about climate change.",
                        metadata={
                            "title": "Climate Change",
                            "url": "https://example.com/climate_change",
                        },
                    ),
                    Document(
                        page_content="Climate change is a significant global challenge.",
                        metadata={
                            "title": "Global Challenge",
                            "url": "https://example.com/global_challenge",
                        },
                    ),
                ],
            ),
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="The title summarizes the main topic of the document.",
                        answer_sentece="## Climate Change",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The context provides a summary of the main topic.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The context provides specific details about the main topic.",
                        answer_sentece="* The main topic is about climate change.",
                        reference_url="https://example.com/climate_change",
                        reference_title="Climate Change",
                    ),
                    AnswerStatement(
                        reasoning="The context provides specific details about the main topic.",
                        answer_sentece="* Climate change is a significant global challenge.",
                        reference_url="https://example.com/global_challenge",
                        reference_title="Global Challenge",
                    ),
                ]
            ),
        ),
        (
            QueryInput(
                question="Who was the emperor of China when the person after whom the oldest extant football team's stadium in Italy is named was 5 years old?",
                context=[
                    Document(
                        page_content="Genoa Cricket and Football Club is the oldest extant football team in Italy. They play at the Stadio Luigi Ferraris.",
                        metadata={
                            "title": "Genoa CFC",
                            "url": "https://en.wikipedia.org/wiki/Genoa_CFC",
                        },
                    ),
                    Document(
                        page_content="Stadio Luigi Ferraris is named after Luigi Ferraris, an Italian footballer and engineer.",
                        metadata={
                            "title": "Stadio Luigi Ferraris",
                            "url": "https://en.wikipedia.org/wiki/Stadio_Luigi_Ferraris",
                        },
                    ),
                    Document(
                        page_content="Luigi Ferraris was born in 1887.",
                        metadata={
                            "title": "Luigi Ferraris",
                            "url": "https://en.wikipedia.org/wiki/Luigi_Ferraris",
                        },
                    ),
                ],
            ),
            AnswerWithCitations(
                statements=[
                    AnswerStatement(
                        reasoning="Question asks for the emperor of China, answer is entitled with his name.",
                        answer_sentece="## Guangxu Emperor",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The context provides a summary of the main topic.",
                        answer_sentece="Here is the what context provides to your question:",
                        reference_url=None,
                        reference_title=None,
                    ),
                    AnswerStatement(
                        reasoning="The oldest extant football team in Italy is Genoa CFC, which plays at the Stadio Luigi Ferraris.",
                        answer_sentece="* The oldest extant football team in Italy is Genoa CFC, which plays at the Stadio Luigi Ferraris.",
                        reference_url="https://en.wikipedia.org/wiki/Genoa_CFC",
                        reference_title="Genoa CFC",
                    ),
                    AnswerStatement(
                        reasoning="Question asks about age of Luigi Ferraris, who is the namesake of the stadium.",
                        answer_sentece="* The stadium is named after Luigi Ferraris, who was born in 1887.",
                        reference_url="https://en.wikipedia.org/wiki/Luigi_Ferraris",
                        reference_title="Luigi Ferraris",
                    ),
                    AnswerStatement(
                        reasoning="As Luigi Ferraris was born in 1887 and was 5 years old in 1892, the emperor of China at that time was the Guangxu Emperor.",
                        answer_sentece="* When Luigi Ferraris was 5 years old, in 1892, the emperor of China was the Guangxu Emperor.",
                        reference_url="https://en.wikipedia.org/wiki/Guangxu_Emperor",
                        reference_title="Guangxu Emperor",
                    ),
                ]
            ),
        ),
    ]


class SubagentInput(BaseModel):
    original_request: str = Field(
        description="The original request or question that initiated the research process."
    )
    research_topic: str = Field(
        description="The specific topic that the subagent is assigned to research."
    )


class SubagentOutput(BaseModel):
    report: str = Field(
        description="A detailed report summarizing the findings and insights gathered by the subagent on its research topic."
    )
    references: list[str] = Field(
        description="A list of URLs that were used as sources for the research conducted by the subagent. If URL is not available, provide the name of the source instead."
    )


# adopted from https://github.com/langchain-ai/deep_research_from_scratch/blob/9efefb9a3411631e773a43cbc3bf911ad2629fac/src/deep_research_from_scratch/prompts.py#L90
subagentPrompt = """You are a research assistant conducting research on the user's input topic using local files.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question.
You can call these tools in series or in parallel, your research strategy is up to you.
</Task>

<Available Tools>
You have access to local search tools, web search tools and thinking tools:
- **retrieve_docs**: Retrieve relevant documents from a local, offline index, try it first
- **tavily_search**: Search the web for relevant information
- **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Do not ask for clarification** - Use the information provided to guide your research or thinking
3. **Start with broader searches** - Use broad, comprehensive queries first
4. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
5. **Execute narrower searches as you gather information** - Fill in the gaps
6. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 1 search tool call maximum
- **Complex queries**: Use up to 3 search tool calls maximum
- **Always stop**: After 3 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 2+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>

<Input>
Research Topic: {research_topic}
</Input>
"""

supervisorPrompt = """You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool.

<Task>
Your focus is to call the "ConductResearch" tool to conduct research against the overall research question passed in by the user. 
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress**
**PARALLEL RESEARCH**: When you identify multiple independent sub-topics that can be explored simultaneously, make multiple ConductResearch tool calls in a single response to enable parallel research execution. This is more efficient than sequential research for comparative or multi-faceted questions. Use at most {max_concurrent_research_units} parallel agents per iteration.
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Do not ask for clarification** - Use the information provided to guide your research or thinking
3. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
4. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless the user request has clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to think_tool and ConductResearch if you cannot find the right sources
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I delegate more research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: List the top 10 coffee shops in San Francisco → Use 1 sub-agent

**Comparisons presented in the user request** can use a sub-agent for each element of the comparison:
- *Example*: Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety → Use 3 sub-agents
- Delegate clear, distinct, non-overlapping subtopics

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather information
- When calling ConductResearch, provide complete standalone instructions - sub-agents can't see other agents' work
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>

<Input>
Topic to research: {research_topic}
</Input>
"""
