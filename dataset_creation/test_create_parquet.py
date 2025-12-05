"""
Test script for create_parquet.py - validates the refactored code without processing all data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from create_parquet import load_titw_metadata, get_test_utterance_ids, filter_samples_by_test, analyze_overlaps

def test_metadata_loading():
    """Test that metadata loading functions work correctly"""
    
    titw_path = Path('../../training/data/titw')
    
    print("Testing metadata loading functions...")
    print("=" * 80)
    
    # Test easy set loading
    print("\n1. Testing easy set (cfg_v3) metadata loading...")
    try:
        easy_samples, easy_spk2utt = load_titw_metadata(titw_path / "bonafide_metadata_cfg_v3")
        print(f"✓ Loaded {len(easy_samples)} samples")
        print(f"✓ Found {len(easy_spk2utt)} speakers")
        
        # Show a sample
        if easy_samples:
            sample = easy_samples[0]
            print(f"\nSample utterance:")
            print(f"  ID: {sample['id']}")
            print(f"  Speaker: {sample['speaker_id']}")
            print(f"  Text: {sample['text'][:50]}...")
            print(f"  Wav path: {sample['wav_path']}")
    except Exception as e:
        print(f"✗ Error loading easy set: {e}")
        return False
    
    # Test hard set loading
    print("\n2. Testing hard set (cfg_v6) metadata loading...")
    try:
        hard_samples, hard_spk2utt = load_titw_metadata(titw_path / "bonafide_metadata_cfg_v6")
        print(f"✓ Loaded {len(hard_samples)} samples")
        print(f"✓ Found {len(hard_spk2utt)} speakers")
    except Exception as e:
        print(f"✗ Error loading hard set: {e}")
        return False
    
    # Test test set loading
    print("\n3. Testing test utterance identification...")
    try:
        test_utt_ids = get_test_utterance_ids(titw_path / "easy" / "test", titw_path / "hard" / "test")
        print(f"✓ Found {len(test_utt_ids)} test utterance IDs")
        
        # Show a sample
        if test_utt_ids:
            sample_id = list(test_utt_ids)[0]
            print(f"\nSample test utterance ID: {sample_id}")
    except Exception as e:
        print(f"✗ Error identifying test utterances: {e}")
        return False
    
    # Test filtering
    print("\n4. Testing sample filtering by test utterances...")
    try:
        easy_subset, easy_test = filter_samples_by_test(easy_samples, test_utt_ids, "Easy")
        hard_subset, hard_test = filter_samples_by_test(hard_samples, test_utt_ids, "Hard")
        print(f"✓ Filtered easy and hard samples")
    except Exception as e:
        print(f"✗ Error filtering samples: {e}")
        return False
    
    # Test overlap analysis
    print("\n4. Testing overlap analysis...")
    assert len(easy_test) == len(hard_test), "Test sets from easy and hard should match in size"
    easy_test.sort(key=lambda x: x['id'])
    hard_test.sort(key=lambda x: x['id'])
    for id1, id2 in zip(easy_test, hard_test):
        assert id1 == id2, "Test utterance IDs should match between easy and hard sets"
    try:
        analyze_overlaps(easy_samples, hard_samples, easy_test)
        print("✓ Overlap analysis completed successfully")
    except Exception as e:
        print(f"✗ Error in overlap analysis: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_metadata_loading()
    sys.exit(0 if success else 1)
