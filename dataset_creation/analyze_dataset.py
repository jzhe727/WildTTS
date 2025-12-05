"""
Analyze TITW dataset to identify shared speakers and audio files between easy and hard sets.
"""

import os
from pathlib import Path
from collections import defaultdict

def load_spk2utt(spk2utt_path):
    """Load speaker-to-utterance mapping"""
    spk_to_utts = {}
    with open(spk2utt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk_id = parts[0]
                utt_ids = parts[1:]
                spk_to_utts[spk_id] = set(utt_ids)
    return spk_to_utts

def load_wav_scp(wav_scp_path):
    """Load utterance-to-wav mapping"""
    utt_to_wav = {}
    with open(wav_scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                utt_to_wav[utt_id] = wav_path
    return utt_to_wav

def get_test_files(test_dir):
    """Get all speakers from test directory"""
    test_speakers = set()
    test_files = {}
    
    for wav_file in Path(test_dir).glob('*.wav'):
        # Extract speaker ID from filename (e.g., id10299-xxx.wav -> id10299)
        filename = wav_file.name
        spk_id = filename.split('-')[0]
        test_speakers.add(spk_id)
        if spk_id not in test_files:
            test_files[spk_id] = []
        test_files[spk_id].append(filename)
    
    return test_speakers, test_files

def analyze_dataset(titw_data_dir='../../training/data/titw'):
    """Analyze TITW dataset for overlaps"""
    
    titw_path = Path(titw_data_dir)
    
    print("=" * 80)
    print("TITW Dataset Overlap Analysis")
    print("=" * 80)
    print()
    
    # Load easy set metadata
    print("Loading easy set (cfg_v3)...")
    easy_spk2utt = load_spk2utt(titw_path / "bonafide_metadata_cfg_v3" / "spk2utt")
    easy_wav_scp = load_wav_scp(titw_path / "bonafide_metadata_cfg_v3" / "wav.scp")
    
    # Load hard set metadata  
    print("Loading hard set (cfg_v6)...")
    hard_spk2utt = load_spk2utt(titw_path / "bonafide_metadata_cfg_v6" / "spk2utt")
    hard_wav_scp = load_wav_scp(titw_path / "bonafide_metadata_cfg_v6" / "wav.scp")
    
    # Get test set files from easy/test (test is shared)
    print("Loading test set...")
    test_speakers, test_files_by_spk = get_test_files(titw_path / "easy" / "test")
    
    print()
    print("-" * 80)
    print("Dataset Statistics")
    print("-" * 80)
    
    # Easy set stats
    easy_speakers = set(easy_spk2utt.keys())
    easy_utts = set(easy_wav_scp.keys())
    print(f"Easy set (cfg_v3):")
    print(f"  Speakers: {len(easy_speakers)}")
    print(f"  Utterances: {len(easy_utts)}")
    
    # Hard set stats
    hard_speakers = set(hard_spk2utt.keys())
    hard_utts = set(hard_wav_scp.keys())
    print(f"\nHard set (cfg_v6):")
    print(f"  Speakers: {len(hard_speakers)}")
    print(f"  Utterances: {len(hard_utts)}")
    
    # Test set stats
    total_test_files = sum(len(files) for files in test_files_by_spk.values())
    print(f"\nTest set:")
    print(f"  Speakers: {len(test_speakers)}")
    print(f"  Audio files: {total_test_files}")
    
    print()
    print("-" * 80)
    print("Overlap Analysis")
    print("-" * 80)
    
    # Shared speakers between easy and hard
    shared_easy_hard_speakers = easy_speakers & hard_speakers
    print(f"\nShared speakers (Easy ∩ Hard): {len(shared_easy_hard_speakers)}")
    print(f"  Percentage of easy speakers: {len(shared_easy_hard_speakers)/len(easy_speakers)*100:.1f}%")
    print(f"  Percentage of hard speakers: {len(shared_easy_hard_speakers)/len(hard_speakers)*100:.1f}%")
    
    # Unique speakers
    easy_only_speakers = easy_speakers - hard_speakers
    hard_only_speakers = hard_speakers - easy_speakers
    print(f"\nUnique to easy set: {len(easy_only_speakers)}")
    print(f"Unique to hard set: {len(hard_only_speakers)}")
    
    # Shared utterances between easy and hard
    shared_utterances = easy_utts & hard_utts
    print(f"\nShared utterances (Easy ∩ Hard): {len(shared_utterances)}")
    print(f"  Percentage of easy utterances: {len(shared_utterances)/len(easy_utts)*100:.1f}%")
    print(f"  Percentage of hard utterances: {len(shared_utterances)/len(hard_utts)*100:.1f}%")
    
    # Unique utterances
    easy_only_utts = easy_utts - hard_utts
    hard_only_utts = hard_utts - easy_utts
    print(f"\nUnique to easy set: {len(easy_only_utts)}")
    print(f"Unique to hard set: {len(hard_only_utts)}")
    
    # Test set overlap with easy and hard
    test_in_easy = test_speakers & easy_speakers
    test_in_hard = test_speakers & hard_speakers
    test_in_both = test_speakers & shared_easy_hard_speakers
    
    print(f"\nTest speakers in easy set: {len(test_in_easy)}")
    print(f"Test speakers in hard set: {len(test_in_hard)}")
    print(f"Test speakers in both easy and hard: {len(test_in_both)}")
    
    # Utterances per speaker statistics
    print()
    print("-" * 80)
    print("Utterances per Speaker Statistics")
    print("-" * 80)
    
    # Easy set stats
    easy_utt_counts = [len(utts) for utts in easy_spk2utt.values()]
    print(f"\nEasy set:")
    print(f"  Maximum utterances per speaker: {max(easy_utt_counts)}")
    print(f"  Minimum utterances per speaker: {min(easy_utt_counts)}")
    print(f"  Average utterances per speaker: {sum(easy_utt_counts)/len(easy_utt_counts):.1f}")
    
    # Hard set stats
    hard_utt_counts = [len(utts) for utts in hard_spk2utt.values()]
    print(f"\nHard set:")
    print(f"  Maximum utterances per speaker: {max(hard_utt_counts)}")
    print(f"  Minimum utterances per speaker: {min(hard_utt_counts)}")
    print(f"  Average utterances per speaker: {sum(hard_utt_counts)/len(hard_utt_counts):.1f}")
    
    # Test set stats
    test_utt_counts = [len(files) for files in test_files_by_spk.values()]
    print(f"\nTest set:")
    print(f"  Maximum utterances per speaker: {max(test_utt_counts)}")
    print(f"  Minimum utterances per speaker: {min(test_utt_counts)}")
    print(f"  Average utterances per speaker: {sum(test_utt_counts)/len(test_utt_counts):.1f}")
    
    # Speaker utterance statistics
    print()
    print("-" * 80)
    print("Utterances per Speaker Statistics")
    print("-" * 80)
    
    # Easy set
    easy_utt_counts = [len(utts) for utts in easy_spk2utt.values()]
    print(f"\nEasy set:")
    print(f"  Min utterances: {min(easy_utt_counts)}")
    print(f"  Max utterances: {max(easy_utt_counts)}")
    print(f"  Average utterances: {sum(easy_utt_counts)/len(easy_utt_counts):.1f}")
    
    # Hard set
    hard_utt_counts = [len(utts) for utts in hard_spk2utt.values()]
    print(f"\nHard set:")
    print(f"  Min utterances: {min(hard_utt_counts)}")
    print(f"  Max utterances: {max(hard_utt_counts)}")
    print(f"  Average utterances: {sum(hard_utt_counts)/len(hard_utt_counts):.1f}")
    
    # Test set
    test_utt_counts = [len(files) for files in test_files_by_spk.values()]
    print(f"\nTest set:")
    print(f"  Min utterances: {min(test_utt_counts)}")
    print(f"  Max utterances: {max(test_utt_counts)}")
    print(f"  Average utterances: {sum(test_utt_counts)/len(test_utt_counts):.1f}")
    
    # Example of shared speakers with utterance counts
    print()
    print("-" * 80)
    print("Example: First 5 Shared Speakers")
    print("-" * 80)
    for i, spk_id in enumerate(sorted(shared_easy_hard_speakers)[:5]):
        easy_count = len(easy_spk2utt[spk_id])
        hard_count = len(hard_spk2utt[spk_id])
        shared_spk_utts = easy_spk2utt[spk_id] & hard_spk2utt[spk_id]
        print(f"\n{spk_id}:")
        print(f"  Easy utterances: {easy_count}")
        print(f"  Hard utterances: {hard_count}")
        print(f"  Shared utterances: {len(shared_spk_utts)}")
        
        # Check if in test set
        if spk_id in test_speakers:
            print(f"  Test files: {len(test_files_by_spk[spk_id])}")
    
    print()
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)
    
    return {
        'easy_speakers': easy_speakers,
        'hard_speakers': hard_speakers,
        'test_speakers': test_speakers,
        'shared_easy_hard_speakers': shared_easy_hard_speakers,
        'easy_utterances': easy_utts,
        'hard_utterances': hard_utts,
        'shared_utterances': shared_utterances,
        'test_files_by_spk': test_files_by_spk
    }

if __name__ == "__main__":
    results = analyze_dataset()
