import asyncio
import asyncpg
import csv
import logging
import os
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"

async def generate_report():


    parser = argparse.ArgumentParser(description="Generate trade reports for the LiveFader bot.")
    parser.add_argument(
        '--full',
        action='store_true',
        help='Generate a full historical report of all closed trades.'
    )
    args = parser.parse_args()

    load_dotenv()
    db_dsn = os.getenv("DATABASE_URL")
    if not db_dsn:
        LOG.error("DATABASE_URL not found in environment or .env file. Cannot proceed.")
        return

    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)


        base_query = "SELECT * FROM positions"


        if args.full:
            report_type = "FULL HISTORICAL"
            today_utc = datetime.now(timezone.utc).date()
            filename = REPORTS_DIR / f"full_trade_history_{today_utc.isoformat()}.csv"

            query = f"{base_query} WHERE status = 'CLOSED' ORDER BY closed_at ASC"
            records = await conn.fetch(query)
        else:
            report_type = "DAILY"
            today_utc = datetime.now(timezone.utc).date()
            report_date = today_utc - timedelta(days=1)
            start_time = datetime(report_date.year, report_date.month, report_date.day, tzinfo=timezone.utc)
            end_time = start_time + timedelta(days=1)
            filename = REPORTS_DIR / f"trade_report_{report_date.isoformat()}.csv"

            query = f"{base_query} WHERE closed_at >= $1 AND closed_at < $2 ORDER BY closed_at ASC"
            records = await conn.fetch(query, start_time, end_time)

        LOG.info(f"Generating {report_type} trade report...")

        if not records:
            LOG.info("No trades found for this report period. No report will be generated.")
            return

        LOG.info(f"Found {len(records)} closed trades to report.")
        REPORTS_DIR.mkdir(exist_ok=True)

        headers = list(records[0].keys())

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for record in records:
                writer.writerow(list(record.values()))

        LOG.info(f"Successfully generated report: {filename}")

    except Exception as e:
        LOG.error(f"An error occurred during report generation: {e}")
    finally:
        if conn:
            await conn.close()
            LOG.info("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(generate_report())
