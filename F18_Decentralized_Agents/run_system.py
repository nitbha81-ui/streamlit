"""Entry point to run the decentralized multi-agent system."""

from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.analyzer_agent import AnalyzerAgent
from core.message_bus import bus

def run_query(query: str):
    print("\n" + "="*60)
    print(f"📝 QUERY: {query}")
    print("="*60)
    
    # Start the conversation
    print("\n🚀 Initiating decentralized agent workflow...")
    
    # Researcher publishes a query request
    researcher = ResearcherAgent()
    researcher.publish("query.request", {
        "type": "research_request",
        "query": query
    })
    
    # Wait for final answer
    print("\n⏳ Agents are working...")
    
    # Collect final answers
    # bus.get_messages returns items shaped as {"topic": ..., "message": {...}},
    # so the actual payload lives under the "message" key.
    final_answers = []
    for record in bus.get_messages("answer.final"):
        payload = record.get("message", {})
        if payload.get("type") == "final_answer":
            final_answers.append(payload)
    
    if final_answers:
        print("\n" + "="*60)
        print("✅ FINAL ANSWER")
        print("="*60)
        print(final_answers[-1]["answer"])
    else:
        print("\n⚠️ No final answer received.")

if __name__ == "__main__":
    # Initialize agents (they automatically subscribe to topics)
    researcher = ResearcherAgent()
    writer = WriterAgent()
    analyzer = AnalyzerAgent()
    
    # Test queries
    test_queries = [
        "What is Ashwagandha coffee?",
        "Tell me about the benefits of turmeric coffee.",
        "Who is Nitin Bhalerao?",
    ]
    
    for query in test_queries:
        run_query(query)