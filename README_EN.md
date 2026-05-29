# deckAi

A Pokemon TCG-style battle simulator with AI-powered card recognition and win-rate based weight learning.

## Features

- **Card Recognition**: Analyze card images using OpenAI Vision API and convert to structured data
- **Battle Simulation**: Run automated battles between multiple decks and collect win-rate statistics
- **Battle Recording & Video**: Record individual matches with state snapshots and generate board visualization videos (MP4)
- **Weight Learning**: Assign weights to decisions (attack selection, energy attachment, retreat/switch) and learn optimal weights from recorded battle outcomes
- **Q-Learning**: Train a neural network (PyTorch) to predict win probability for state-action pairs

## Tech Stack

- Python 3.10+
- OpenAI Vision API (card image analysis)
- PyTorch (Q-learning model)
- FFmpeg (video generation)
- pytest (testing)

## Architecture

```
Card Images → OpenAI Vision API → JSON → card/data.py
                                            ↓
                                    Battle Simulation
                                            ↓
                                    Record & Learn Weights
                                            ↓
                                    Improved AI Decisions
```

## Project Structure

```
deckAi/
├── card/                  # Card definitions & master data
│   ├── model.py           # Types (PokemonCard, Attack, etc.)
│   └── data.py            # Card data (partially generated from JSON)
├── game/                  # Game state, turn progression, AI decisions
│   ├── state.py           # Game state, setup, helpers
│   ├── damage.py          # Damage calculation
│   ├── evolution.py       # Evolution logic
│   ├── trainers.py        # Trainer cards (Items, Supporters, Tools)
│   ├── attack.py          # Attack execution
│   ├── turn.py            # Turn progression
│   └── weights.py         # Decision weights (learning)
├── scripts/               # CLI tools
│   ├── simulate.py        # Run N battles, show win rates
│   ├── record_game.py     # Record a single match
│   ├── make_video.py      # Generate MP4 from recorded match
│   ├── train_weights.py   # Learn weights from battle records
│   └── train_q_torch.py   # Train Q-network with PyTorch
├── tests/                 # pytest test suite
├── board_render.py        # Board state visualization
├── read_cards.py          # Card image → JSON pipeline
└── update_cards_from_json.py  # JSON → card/data.py updater
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simulation (1000 battles, Deck 0 vs Deck 1)
python scripts/simulate.py

# Record a match and generate video
python scripts/record_game.py
python scripts/make_video.py

# Train weights from recorded battles
python scripts/train_weights.py

# Run with learned weights
python scripts/simulate.py 100 --weights weights/weights.json
```

## Card Recognition

Requires OpenAI API key in `.env`:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Scan card images
python read_cards.py

# Or use a deck code
python read_cards.py gLn9ng-Cr455W-iNnLLN

# Update card/data.py with scanned results
python update_cards_from_json.py
```

## Testing

```bash
python -m pytest tests/ -v
```

## [Japanese README (日本語)](README.md)
