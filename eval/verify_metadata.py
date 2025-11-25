#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification script for TITW test set metadata.
Checks that all metadata files are present and consistent.
"""

import sys
from pathlib import Path


def verify_metadata(metadata_dir):
    """Verify metadata files exist and are consistent."""
    
    metadata_dir = Path(metadata_dir)
    
    # Check required files exist
    required_files = [
        'text_test',
        'wav_test.scp',
        'utt2spk_test',
        'spk2utt_test'
    ]
    
    print("Checking required files...")
    for filename in required_files:
        filepath = metadata_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            return False
    
    # Load and count utterances from each file
    print("\nLoading metadata...")
    
    # text_test
    text_utts = set()
    with open(metadata_dir / 'text_test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                utt_id = line.split()[0]
                text_utts.add(utt_id)
    print(f"  text_test: {len(text_utts)} utterances")
    
    # wav_test.scp
    wav_utts = set()
    with open(metadata_dir / 'wav_test.scp', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                utt_id = line.split()[0]
                wav_utts.add(utt_id)
    print(f"  wav_test.scp: {len(wav_utts)} utterances")
    
    # utt2spk_test
    utt2spk = {}
    with open(metadata_dir / 'utt2spk_test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    utt_id, spk_id = parts
                    utt2spk[utt_id] = spk_id
    print(f"  utt2spk_test: {len(utt2spk)} utterances")
    
    # spk2utt_test
    spk2utt = {}
    with open(metadata_dir / 'spk2utt_test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    spk_id = parts[0]
                    utts = parts[1:]
                    spk2utt[spk_id] = utts
    print(f"  spk2utt_test: {len(spk2utt)} speakers")
    
    # Check consistency
    print("\nChecking consistency...")
    
    # All files should have same utterances
    if text_utts == wav_utts == set(utt2spk.keys()):
        print(f"  ✓ All files have {len(text_utts)} utterances")
    else:
        print("  ✗ Utterance mismatch across files!")
        print(f"    text_test: {len(text_utts)}")
        print(f"    wav_test.scp: {len(wav_utts)}")
        print(f"    utt2spk_test: {len(utt2spk)}")
        return False
    
    # Check speaker counts
    spk_ids_from_utt2spk = set(utt2spk.values())
    spk_ids_from_spk2utt = set(spk2utt.keys())
    
    if spk_ids_from_utt2spk == spk_ids_from_spk2utt:
        print(f"  ✓ Consistent speaker IDs: {len(spk_ids_from_utt2spk)} speakers")
    else:
        print("  ✗ Speaker ID mismatch!")
        return False
    
    # Show speaker statistics
    print("\nSpeaker statistics:")
    utts_per_spk = [len(utts) for utts in spk2utt.values()]
    print(f"  Total speakers: {len(spk2utt)}")
    print(f"  Min utterances per speaker: {min(utts_per_spk)}")
    print(f"  Max utterances per speaker: {max(utts_per_spk)}")
    print(f"  Average utterances per speaker: {sum(utts_per_spk) / len(utts_per_spk):.1f}")
    
    # Sample data
    print("\nSample entries:")
    sample_utt = list(text_utts)[0]
    
    # Get text
    with open(metadata_dir / 'text_test', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(sample_utt):
                text = line.split(' ', 1)[1].strip()
                break
    
    # Get wav path
    with open(metadata_dir / 'wav_test.scp', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(sample_utt):
                wav_path = line.split(' ', 1)[1].strip()
                break
    
    # Get speaker
    spk_id = utt2spk[sample_utt]
    
    print(f"  Utterance ID: {sample_utt}")
    print(f"  Speaker ID: {spk_id}")
    print(f"  Wav path: {wav_path}")
    print(f"  Text: {text}")
    
    print("\n" + "="*60)
    print("✓ Metadata verification successful!")
    print("="*60)
    
    return True


def main():
    if len(sys.argv) > 1:
        metadata_dir = sys.argv[1]
    else:
        # Default path relative to script
        script_dir = Path(__file__).resolve().parent
        metadata_dir = script_dir / 'titw-test' / 'metadata'
    
    print("="*60)
    print("TITW Test Set Metadata Verification")
    print("="*60)
    print(f"Metadata directory: {metadata_dir}")
    print("="*60)
    print()
    
    if not Path(metadata_dir).exists():
        print(f"Error: Directory not found: {metadata_dir}")
        sys.exit(1)
    
    success = verify_metadata(metadata_dir)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
