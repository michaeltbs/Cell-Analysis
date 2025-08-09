import os
import argparse
from src.analysis import create_summary_csv

def main():
	parser = argparse.ArgumentParser(description="Berechnet Mittelwerte und Summary aus Master-CSV")
	parser.add_argument("--input", type=str, default="results/All_Counts_Master.csv",
		help="Pfad zur Master-CSV")
	parser.add_argument("--output", type=str, default="analysis_results",
		help="Basis-Name für die Ausgabedateien")
	parser.add_argument("--no-md", action="store_true", help="Keine Markdown-Zusammenfassung schreiben")
	args = parser.parse_args()

	if not os.path.exists(args.input):
		print(f"❌ Eingabedatei nicht gefunden: {args.input}")
		return

	create_summary_csv(args.input, args.output, write_markdown=(not args.no_md))
	print("✅ Fertig")

if __name__ == "__main__":
	main()