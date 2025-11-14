#!/usr/bin/env python3
"""
Simple Survey Example

Demonstrates basic PsychScanner usage with a simple questionnaire.
"""

from psychscanner import ExpCard, ScannerModel

def main():
    """Run a simple survey experiment."""
    
    # Configure experiment
    exp_card = ExpCard(
        model="gpt-3.5-turbo",      # Use GPT-3.5
        family="openai",             # OpenAI provider
        task_file=None,              # Use default VVIQ survey
        memory="SingleTurn",         # Independent trials
        projectname="simple_survey", # Results folder
        enabletqdm=True              # Show progress
    )
    
    print("Starting simple survey experiment...")
    print(f"Model: {exp_card.card_in.model}")
    print(f"Task: Default VVIQ survey")
    print(f"Results will be saved to: {exp_card.data_root_dir}")
    
    # Create scanner and run
    scanner = ScannerModel(exp_card)
    results = scanner.run(progress_bar=True)
    
    # Summary
    print(f"\nCompleted! {len(results[0])} trials finished.")
    print(f"Results saved to: {scanner.data_root_dir}")
    
    return results


if __name__ == "__main__":
    main()
