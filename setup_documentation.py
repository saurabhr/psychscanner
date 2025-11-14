#!/usr/bin/env python3
"""
PsychScanner Documentation Setup Script

This script creates the complete documentation structure for PsychScanner.
Run this from the root of your psychscanner repository.

Usage:
    python setup_documentation.py [--dry-run]

Options:
    --dry-run    Show what would be created without actually creating files
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


# Define the documentation structure
DOC_STRUCTURE = {
    'root_files': [
        'CONTRIBUTING.md',
        'CHANGELOG.md',
    ],
    'docs': {
        'files': [
            'quickstart.md',
            'configuration.md',
            'deployment.md',
            'troubleshooting.md',
        ],
        'api': [
            'index.md',
            'expcard.md',
            'scanner_model.md',
            'session_tunnel.md',
            'parsers.md',
        ],
        'guides': [
            'index.md',
            'survey_tasks.md',
            'cognitive_tasks.md',
            'custom_parsers.md',
            'memory_types.md',
            'session_recovery.md',
        ],
        'examples': [
            'index.md',
            'basic_survey.md',
            'multi_persona.md',
            'reality_monitoring.md',
            'feedback_loop.md',
        ],
    },
    'examples': {
        'basic': [
            'simple_survey.py',
            'README.md',
        ],
        'advanced': [
            'multi_persona.py',
            'reality_monitoring.py',
            'feedback_system.py',
            'README.md',
        ],
        'tasks': [
            'example_survey.json',
            'example_cognitive.json',
            'README.md',
        ],
    },
}


# File templates
FILE_TEMPLATES = {
    'docs/api/index.md': """# API Reference

Complete API documentation for PsychScanner.

## Core Classes

- [ExpCard](expcard.md) - Experiment configuration
- [ScannerModel](scanner_model.md) - Main scanning engine
- [SessionTunnel](session_tunnel.md) - Session management
- [Parsers](parsers.md) - Response parsers

## Quick Links

- [Installation](../installation.md)
- [Quick Start](../quickstart.md)
- [Examples](../examples/)
""",

    'docs/guides/index.md': """# User Guides

Comprehensive guides for using PsychScanner.

## Getting Started

- [Survey Tasks](survey_tasks.md) - Creating and running surveys
- [Cognitive Tasks](cognitive_tasks.md) - Implementing cognitive experiments
- [Custom Parsers](custom_parsers.md) - Building response parsers
- [Memory Types](memory_types.md) - Understanding memory management
- [Session Recovery](session_recovery.md) - Checkpoint and resume experiments

## Advanced Topics

Coming soon:
- Multi-agent experiments
- Real-time monitoring
- Custom task types
- Integration with analysis pipelines
""",

    'docs/examples/index.md': """# Examples

Practical examples demonstrating PsychScanner capabilities.

## Basic Examples

- [Simple Survey](basic_survey.md) - Running a basic questionnaire
- See `examples/basic/` for code

## Advanced Examples

- [Multi-Persona Study](multi_persona.md) - Testing multiple personas
- [Reality Monitoring Task](reality_monitoring.md) - Complex cognitive task
- [Feedback Loop System](feedback_loop.md) - Trial-by-trial feedback

## Running Examples

```bash
# Clone the repository
git clone https://github.com/saurabhr/psychscanner.git
cd psychscanner

# Run basic example
python examples/basic/simple_survey.py

# Run advanced example
python examples/advanced/multi_persona.py
```

## Contributing Examples

Have an interesting use case? [Contribute your example](../../CONTRIBUTING.md)!
""",

    'examples/basic/README.md': """# Basic Examples

Simple examples to get started with PsychScanner.

## Files

- `simple_survey.py` - Basic survey implementation

## Running

```bash
# Install psychscanner
pip install psychscanner

# Run the example
python simple_survey.py
```

## What You'll Learn

- Creating an ExpCard
- Running a basic survey
- Viewing results
- Using different LLM providers
""",

    'examples/advanced/README.md': """# Advanced Examples

Complex examples demonstrating advanced features.

## Files

- `multi_persona.py` - Multi-persona study
- `reality_monitoring.py` - Reality monitoring task
- `feedback_system.py` - Custom feedback implementation

## Prerequisites

```bash
pip install psychscanner
export OPENAI_API_KEY="your-key"  # or other provider
```

## Running

```bash
python multi_persona.py
python reality_monitoring.py
python feedback_system.py
```

## Topics Covered

- Multiple personas
- Conversational memory
- Custom parsers
- Feedback systems
- Session recovery
- Complex task structures
""",

    'examples/tasks/README.md': """# Example Task Definitions

Sample task JSON files for PsychScanner.

## Files

- `example_survey.json` - Simple survey template
- `example_cognitive.json` - Cognitive task template

## Structure

### Survey Task
```json
{
    "tasktype": "survey",
    "taskname": "your_survey",
    "instructions": {...},
    "items": {...},
    "parser": "ParserName"
}
```

### Cognitive Task
```json
{
    "tasktype": "sc",
    "taskname": "your_task",
    "instructions": {...},
    "items": {...},
    "chain_type": "trial"
}
```

## Using These Templates

```python
from psychscanner import ExpCard, ScannerModel

exp_card = ExpCard(
    model="gpt-3.5-turbo",
    family="openai",
    task_file="examples/tasks/example_survey.json",
    projectname="my_study"
)

scanner = ScannerModel(exp_card)
results = scanner.run()
```

## Creating Your Own

1. Copy a template
2. Modify instructions and items
3. Choose appropriate parser
4. Test with mock model first

See [Task Guide](../../docs/guides/survey_tasks.md) for details.
""",

    'examples/basic/simple_survey.py': """#!/usr/bin/env python3
\"\"\"
Simple Survey Example

Demonstrates basic PsychScanner usage with a simple questionnaire.
\"\"\"

from psychscanner import ExpCard, ScannerModel

def main():
    \"\"\"Run a simple survey experiment.\"\"\"
    
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
    print(f"\\nCompleted! {len(results[0])} trials finished.")
    print(f"Results saved to: {scanner.data_root_dir}")
    
    return results


if __name__ == "__main__":
    main()
""",

    'examples/tasks/example_survey.json': """{
    "tasktype": "survey",
    "taskname": "example_survey",
    "instructions": {
        "definition": [
            "You will rate your agreement with the following statements.",
            "Use a scale from 1 (strongly disagree) to 5 (strongly agree).",
            "Be honest and thoughtful in your responses."
        ]
    },
    "contexts": [
        "Openness to Experience",
        "Conscientiousness"
    ],
    "contexts_id": ["O", "C"],
    "context_present": true,
    "items": {
        "O_1": "I enjoy trying new things and exploring new ideas.",
        "O_2": "I am comfortable with ambiguity and uncertainty.",
        "O_3": "I prefer routine and predictability.",
        "C_1": "I am organized and pay attention to details.",
        "C_2": "I follow through on my commitments.",
        "C_3": "I tend to procrastinate on important tasks."
    },
    "parser": "DefaultLiteralAgree",
    "trial_chain": "items"
}
""",

    '.readthedocs.yaml': """# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html

version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.13"

sphinx:
  configuration: docs/conf.py
  fail_on_warning: false

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs

formats:
  - pdf
  - epub
""",
}


def create_directory_structure(base_path: Path, structure: Dict, dry_run: bool = False) -> List[Path]:
    """Create directory structure from nested dictionary.
    
    Args:
        base_path: Base directory path
        structure: Nested dictionary defining structure
        dry_run: If True, only print what would be created
    
    Returns:
        List of created directories
    """
    created_dirs = []
    
    for key, value in structure.items():
        if key == 'files':
            # These are files, not directories
            continue
            
        dir_path = base_path / key
        
        if isinstance(value, dict):
            # This is a subdirectory with more structure
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
            print(f"  📁 {dir_path.relative_to(Path.cwd())}")
            
            # Recursively create subdirectories
            created_dirs.extend(create_directory_structure(dir_path, value, dry_run))
        elif isinstance(value, list):
            # This is a directory with files
            if not dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
            print(f"  📁 {dir_path.relative_to(Path.cwd())}")
    
    return created_dirs


def create_file(file_path: Path, content: str, dry_run: bool = False) -> bool:
    """Create a file with given content.
    
    Args:
        file_path: Path to file
        content: File content
        dry_run: If True, only print what would be created
    
    Returns:
        True if file was created/updated, False if skipped
    """
    relative_path = file_path.relative_to(Path.cwd())
    
    if file_path.exists() and not dry_run:
        # File exists - ask before overwriting
        response = input(f"  ⚠️  {relative_path} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print(f"  ⏭️  Skipped {relative_path}")
            return False
    
    if not dry_run:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        print(f"  ✅ Created {relative_path}")
    else:
        print(f"  📄 Would create {relative_path}")
    
    return True


def create_placeholder_file(file_path: Path, dry_run: bool = False) -> bool:
    """Create a placeholder markdown file.
    
    Args:
        file_path: Path to file
        dry_run: If True, only print what would be created
    
    Returns:
        True if created, False if skipped
    """
    title = file_path.stem.replace('_', ' ').title()
    content = f"""# {title}

*This documentation is under development.*

## Coming Soon

This section will contain detailed information about {title.lower()}.

## Placeholder

For now, please refer to:
- [Quick Start Guide](../quickstart.md)
- [API Reference](../api/)
- [Examples](../examples/)

## Contributing

Help us improve this documentation! See [Contributing Guide](../../CONTRIBUTING.md).
"""
    
    return create_file(file_path, content, dry_run)


def setup_documentation(dry_run: bool = False) -> Tuple[int, int]:
    """Set up complete documentation structure.
    
    Args:
        dry_run: If True, only show what would be created
    
    Returns:
        Tuple of (files_created, files_skipped)
    """
    base_path = Path.cwd()
    files_created = 0
    files_skipped = 0
    
    print("\n🚀 Setting up PsychScanner documentation...\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be created\n")
    
    # Create directory structure
    print("📁 Creating directories...")
    
    # Create docs structure
    docs_path = base_path / 'docs'
    create_directory_structure(docs_path, DOC_STRUCTURE['docs'], dry_run)
    
    # Create examples structure  
    examples_path = base_path / 'examples'
    create_directory_structure(examples_path, DOC_STRUCTURE['examples'], dry_run)
    
    print("\n📄 Creating files...")
    
    # Create files with templates
    for file_path_str, content in FILE_TEMPLATES.items():
        file_path = base_path / file_path_str
        if create_file(file_path, content, dry_run):
            files_created += 1
        else:
            files_skipped += 1
    
    # Create placeholder files for docs
    print("\n📝 Creating placeholder documentation files...")
    
    placeholder_files = [
        'docs/configuration.md',
        'docs/deployment.md',
        'docs/troubleshooting.md',
        'docs/api/expcard.md',
        'docs/api/scanner_model.md',
        'docs/api/session_tunnel.md',
        'docs/api/parsers.md',
        'docs/guides/survey_tasks.md',
        'docs/guides/cognitive_tasks.md',
        'docs/guides/custom_parsers.md',
        'docs/guides/memory_types.md',
        'docs/guides/session_recovery.md',
        'docs/examples/basic_survey.md',
        'docs/examples/multi_persona.md',
        'docs/examples/reality_monitoring.md',
        'docs/examples/feedback_loop.md',
    ]
    
    for file_path_str in placeholder_files:
        file_path = base_path / file_path_str
        if create_placeholder_file(file_path, dry_run):
            files_created += 1
        else:
            files_skipped += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"  ✅ Files created: {files_created}")
    print(f"  ⏭️  Files skipped: {files_skipped}")
    
    if not dry_run:
        print(f"\n  📂 Documentation root: {docs_path}")
        print(f"  📂 Examples root: {examples_path}")
        print("\n🎉 Documentation structure created successfully!")
        print("\n📚 Next steps:")
        print("  1. Fill in placeholder documentation files")
        print("  2. Build documentation: cd docs && make html")
        print("  3. View locally: python -m http.server 8000 -d docs/_build/html")
        print("  4. Commit changes: git add . && git commit -m 'Add documentation'")
    else:
        print("\n💡 Run without --dry-run to create files")
    
    return files_created, files_skipped


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Set up PsychScanner documentation structure'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without creating files'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not (Path.cwd() / 'pyproject.toml').exists():
        print("❌ Error: pyproject.toml not found!")
        print("   Please run this script from the psychscanner root directory.")
        sys.exit(1)
    
    try:
        setup_documentation(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()