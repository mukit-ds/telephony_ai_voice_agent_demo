import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool
)
from livekit.plugins import openai, cartesia, silero

load_dotenv()
logger = logging.getLogger("telephony-agent")

@function_tool
async def get_current_time() -> str:
    """Get the current time."""
    return f"The current time is {datetime.now().strftime('%I:%M %p')}"

async def entrypoint(ctx: JobContext):
    """Main entry point for the telephony voice agent."""
    try:
        await ctx.connect()
        logger.info("Connected to LiveKit server")

        # Wait for participant with timeout
        try:
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(),
                timeout=30.0
            )
            logger.info(f"Call connected from: {participant.identity}")
        except asyncio.TimeoutError:
            logger.error("No call received within 30 seconds")
            return

        # Initialize the conversational agent
        agent = Agent(
            instructions="""You are a friendly and helpful AI assistant answering phone calls. 

            Your personality:
            - Professional yet warm and approachable
            - Speak clearly and at a moderate pace for phone calls
            - Keep responses concise but complete
            - Ask clarifying questions when needed

            Your capabilities:
            - Answer questions on a wide range of topics
            - Provide weather information when asked
            - Tell the current time
            - Have natural conversations

            Always identify yourself as an AI assistant when asked.
            Keep responses conversational and under 30 seconds for phone clarity.""",
            tools=[get_current_time]
        )

        # Configure the voice processing pipeline
        session = AgentSession(
            vad=silero.VAD.load(),  # Voice Activity Detection
            stt=openai.STT(
                model="whisper-1",
                language="en"
                # Remove unsupported parameters: suppress_blank, word_timestamps
            ),
            llm=openai.LLM(
                model="gpt-4o-mini",
                temperature=0.7
            ),
            tts=cartesia.TTS(
                model="sonic-2",
                voice="a0e99841-438c-4a64-b679-ae501e7d6091",
                language="en",
                speed=1.0,
                sample_rate=24000
            )
        )

        # Start the agent session
        await session.start(agent=agent, room=ctx.room)
        logger.info("Agent session started")

        # Generate time-based greeting
        hour = datetime.now().hour
        time_greeting = (
            "Good morning" if hour < 12 else
            "Good afternoon" if hour < 18 else
            "Good evening"
        )

        await session.generate_reply(
            instructions=f"""Say '{time_greeting}! Thank you for calling. How can I help you today?'
            Speak warmly and professionally at a moderate pace."""
        )

        logger.info("Initial greeting delivered")

    except Exception as e:
        logger.error(f"Error in agent operation: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the agent
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="telephony_agent"
    ))