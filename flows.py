from crewai.flow import Flow, start, listen
from pydantic import BaseModel

from crewai import Crew, Process
from agents import website_analyst, competitor_analyst, competitor_researcher, report_writer
from task import analysis_task, competitor_finding_task, competitor_researching_task, writing_task


# ---- State Model ----
class SEOState(BaseModel):
    website: str
    analysis: str | None = None
    competitors: list[str] | None = None
    research: str | None = None
    report: str | None = None


# ---- Flow ----
class SEOAnalysisFlow(Flow[SEOState]):

    @start()
    def run_analysis(self) -> str:
        """Start by analyzing the website with the analyst agent."""
        crew = Crew(
            agents=[website_analyst],
            tasks=[analysis_task],
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={"website": self.state.website})
        self.state.analysis = result
        return result

    @listen(run_analysis)
    def find_competitors(self, _: str) -> list[str]:
        """After analysis, find competitors."""
        crew = Crew(
            agents=[competitor_analyst],
            tasks=[competitor_finding_task],
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={"website": self.state.website})
        self.state.competitors = result if isinstance(result, list) else [result]
        return self.state.competitors

    @listen(find_competitors)
    def research_competitors(self, _: list[str]) -> str:
        """After finding competitors, research them."""
        crew = Crew(
            agents=[competitor_researcher],
            tasks=[competitor_researching_task],
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={"website": self.state.website})
        self.state.research = result
        return result

    @listen(research_competitors)
    def write_report(self, _: str) -> str:
        """Finally, synthesize everything into a report."""
        crew = Crew(
            agents=[report_writer],
            tasks=[writing_task],
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={"website": self.state.website})
        self.state.report = result
        return result
