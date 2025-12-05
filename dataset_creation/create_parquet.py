import os
import json
from pathlib import Path
from datasets import Dataset, Audio, Features, Value
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

def load_titw_metadata(metadata_dir):
    """Load metadata similar to eval/generate_titw.py"""
    metadata_dir = Path(metadata_dir)
    
    # Load text
    text_dict = {}
    with open(metadata_dir / 'text', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, text = parts
                text_dict[utt_id] = text
    
    # Load wav.scp
    wav_dict = {}
    with open(metadata_dir / 'wav.scp', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                wav_dict[utt_id] = wav_path
    
    # Load utt2spk
    spk_dict = {}
    with open(metadata_dir / 'utt2spk', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                utt_id, spk_id = parts
                spk_dict[utt_id] = spk_id
    
    # Load spk2utt for speaker-based sharding
    spk2utt = {}
    with open(metadata_dir / 'spk2utt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk_id = parts[0]
                utt_ids = parts[1:]
                spk2utt[spk_id] = utt_ids
    
    # Combine metadata
    samples = []
    for utt_id in text_dict.keys():
        if utt_id in wav_dict and utt_id in spk_dict:
            samples.append({
                'id': utt_id,
                'text': text_dict[utt_id],
                'wav_path': wav_dict[utt_id],
                'speaker_id': spk_dict[utt_id]
            })
    
    return samples, spk2utt

def get_test_utterance_ids(easy_test_dir, hard_test_dir):
    """Get set of test utterance IDs from test directories"""
    test_utt_ids = set()
    
    # Get from easy/test
    easy_test_path = Path(easy_test_dir)
    if easy_test_path.exists():
        for wav_file in easy_test_path.glob('*.wav'):
            test_utt_ids.add(wav_file.stem)
    
    # Get from hard/test  
    hard_test_path = Path(hard_test_dir)
    if hard_test_path.exists():
        for wav_file in hard_test_path.glob('*.wav'):
            test_utt_ids.add(wav_file.stem)
    
    return test_utt_ids

def filter_samples_by_test(samples, test_utt_ids, subset_name):
    """Split samples into subset and test based on test utterance IDs"""
    subset_samples = []
    test_samples = []
    
    for sample in samples:
        if sample['id'] in test_utt_ids:
            test_samples.append(sample)
        else:
            subset_samples.append(sample)
    
    print(f"  {subset_name}: {len(subset_samples)} subset samples, {len(test_samples)} test samples")
    return subset_samples, test_samples

def analyze_overlaps(easy_subset, hard_subset, test_samples):
    """Analyze and report overlaps between subsets"""
    easy_utts = set(s['id'] for s in easy_subset)
    hard_utts = set(s['id'] for s in hard_subset)
    test_utts = set(s['id'] for s in test_samples)
    
    easy_speakers = set(s['speaker_id'] for s in easy_subset)
    hard_speakers = set(s['speaker_id'] for s in hard_subset)
    test_speakers = set(s['speaker_id'] for s in test_samples)
    
    shared_utts_easy_hard = easy_utts & hard_utts
    shared_speakers_easy_hard = easy_speakers & hard_speakers
    
    print("\n" + "=" * 80)
    print("Dataset Overlap Report")
    print("=" * 80)
    print(f"\nSpeakers:")
    print(f"  Easy subset: {len(easy_speakers)}")
    print(f"  Hard subset: {len(hard_speakers)}")
    print(f"  Test subset: {len(test_speakers)}")
    print(f"  Shared (Easy ∩ Hard): {len(shared_speakers_easy_hard)}")
    
    print(f"\nUtterances:")
    print(f"  Easy subset: {len(easy_utts)}")
    print(f"  Hard subset: {len(hard_utts)}")
    print(f"  Test subset: {len(test_utts)}")
    print(f"  Shared utterance IDs (Easy ∩ Hard): {len(shared_utts_easy_hard)}")
    
    print(f"\nNote: Test utterances excluded from easy and hard subsets.")
    print(f"      Easy and hard have different audio despite some shared utterance IDs.")
    print("=" * 80 + "\n")

def create_dataset_with_speaker_shards(samples, titw_path, subset_name, output_path, features, max_speakers_per_shard=10):
    """
    Create sharded dataset where each shard contains samples from one or more speakers.
    This enables:
    - Efficient speaker filtering without loading entire dataset
    - Parallel dataloader workers
    - Streaming with speaker-based sampling
    
    Args:
        samples: List of sample dictionaries
        titw_path: Path to TITW dataset
        subset_name: Name of the subset (easy, hard, test)
        output_path: Output directory
        features: HuggingFace Features schema
        max_speakers_per_shard: Maximum number of speakers per shard file
    
    Returns:
        Set of speaker IDs
    """
    
    # Group samples by speaker
    speaker_samples = defaultdict(list)
    for sample in samples:
        speaker_samples[sample['speaker_id']].append(sample)
    
    # Create shards
    speakers = sorted(speaker_samples.keys())
    speaker_to_shard = {}
    shard_metadata = []
    
    subset_output_path = output_path / subset_name
    subset_output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate total number of shards
    num_shards = (len(speakers) + max_speakers_per_shard - 1) // max_speakers_per_shard
    
    print(f"\nCreating {num_shards} shards for {subset_name} ({len(speakers)} speakers)")
    
    shard_id = 0
    
    # Group speakers into shards
    for i in range(0, len(speakers), max_speakers_per_shard):
        shard_speakers = speakers[i:i+max_speakers_per_shard]
        shard_data = []
        
        for spk_id in shard_speakers:
            speaker_to_shard[spk_id] = shard_id
            
            for sample in speaker_samples[spk_id]:
                wav_full_path = titw_path 
                if subset_name == 'test':
                    wav_full_path = wav_full_path / "easy" / sample['wav_path']
                else:
                    wav_full_path = wav_full_path / subset_name / sample['wav_path']
                
                if not wav_full_path.exists():
                    print(f"Warning: {wav_full_path} not found")
                    continue
                
                # Read audio file
                audio_array, sample_rate = sf.read(wav_full_path)
                
                shard_data.append({
                    'id': sample['id'],
                    'audio': {
                        'array': audio_array,
                        'sampling_rate': sample_rate,
                        'path': str(wav_full_path)
                    },
                    'text': sample['text'],
                    'speaker_id': spk_id
                })
        
        if shard_data:
            # Create and save shard
            shard_dataset = Dataset.from_list(shard_data, features=features)
            shard_file = subset_output_path / f"data-{shard_id:05d}-of-{num_shards:05d}.parquet"
            shard_dataset.to_parquet(shard_file)
            
            # Store shard metadata
            shard_metadata.append({
                'shard_id': shard_id,
                'filename': shard_file.name,
                'num_samples': len(shard_data),
                'speakers': shard_speakers,
                'num_speakers': len(shard_speakers)
            })
            
            print(f"  Shard {shard_id:05d}: {len(shard_data)} samples from {len(shard_speakers)} speakers -> {shard_file.name}")
            shard_id += 1
    
    # Save speaker-to-shard mapping
    mapping_file = subset_output_path / "speaker_to_shard.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            'speaker_to_shard': speaker_to_shard,
            'shard_metadata': shard_metadata,
            'total_speakers': len(speakers),
            'total_shards': num_shards
        }, f, indent=2)
    
    print(f"  Saved speaker-to-shard mapping: {mapping_file}")
    
    return set(speakers)

def create_parquet_dataset(
    titw_data_dir='training/data/titw',
    output_dir='training/data/titw_hf'
):
    """
    Create speaker-class Parquet dataset compatible with HuggingFace datasets.
    
    Creates three main subsets:
    - easy: From easy/bonafide_metadata_cfg_v3 (excluding test utterances)
    - hard: From hard/bonafide_metadata_cfg_v6 (excluding test utterances)
    - test: Utterances from easy/test and hard/test directories
    
    Each subset has speaker_id as the class label.
    Test utterances are excluded from easy and hard subsets.
    """
    
    titw_path = Path(titw_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Creating TITW HuggingFace Dataset with Speaker Classes")
    print("=" * 80)
    
    # Load metadata for easy and hard
    print("\nLoading metadata...")
    easy_all_samples, easy_spk2utt = load_titw_metadata(titw_path / "bonafide_metadata_cfg_v3")
    print(f"  Easy (all): {len(easy_all_samples)} utterances")
    
    hard_all_samples, hard_spk2utt = load_titw_metadata(titw_path / "bonafide_metadata_cfg_v6")
    print(f"  Hard (all): {len(hard_all_samples)} utterances")
    
    # Get test utterance IDs from test directories
    print("\nIdentifying test utterances...")
    test_utt_ids = get_test_utterance_ids(titw_path / "easy" / "test", titw_path / "hard" / "test")
    print(f"  Found {len(test_utt_ids)} unique test utterances")
    
    # Split easy and hard into subset and test
    print("\nSplitting into subsets...")
    easy_subset, easy_test = filter_samples_by_test(easy_all_samples, test_utt_ids, "Easy")
    hard_subset, hard_test = filter_samples_by_test(hard_all_samples, test_utt_ids, "Hard")
    
    # Use easy_test as the test samples (they should be the same in both)
    test_samples = easy_test
    
    # Analyze and report overlaps
    analyze_overlaps(easy_subset, hard_subset, test_samples)
    
    # Define schema
    features = Features({
        'id': Value('string'),
        'audio': Audio(sampling_rate=16000),  # TITW uses 16kHz
        'text': Value('string'),
        'speaker_id': Value('string')  # This is the class label
    })
    
    # Create datasets with speaker shards
    print("\nCreating sharded datasets with speaker classes...")
    
    easy_speakers = create_dataset_with_speaker_shards(
        easy_subset, titw_path, "easy", output_path, features
    )
    
    hard_speakers = create_dataset_with_speaker_shards(
        hard_subset, titw_path, "hard", output_path, features
    )
    
    test_speakers = create_dataset_with_speaker_shards(
        test_samples, titw_path, "test", output_path, features
    )
    
    # Create a summary file
    summary_path = output_path / "dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("TITW HuggingFace Dataset Summary (Sharded)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Dataset Structure: Sharded by speaker for efficient streaming and filtering\n\n")
        f.write(f"Easy subset (training data from easy source):\n")
        f.write(f"  Speakers (classes): {len(easy_speakers)}\n")
        f.write(f"  Utterances: {len(easy_subset)}\n\n")
        f.write(f"Hard subset (training data from hard source):\n")
        f.write(f"  Speakers (classes): {len(hard_speakers)}\n")
        f.write(f"  Utterances: {len(hard_subset)}\n\n")
        f.write(f"Test subset (evaluation data, shared between easy and hard):\n")
        f.write(f"  Speakers (classes): {len(test_speakers)}\n")
        f.write(f"  Utterances: {len(test_samples)}\n\n")
        f.write("Directory structure:\n")
        f.write(f"  {output_dir}/easy/data-00000-of-XXXXX.parquet (multiple shards)\n")
        f.write(f"  {output_dir}/easy/speaker_to_shard.json\n")
        f.write(f"  {output_dir}/hard/data-00000-of-XXXXX.parquet (multiple shards)\n")
        f.write(f"  {output_dir}/hard/speaker_to_shard.json\n")
        f.write(f"  {output_dir}/test/data-00000-of-XXXXX.parquet (multiple shards)\n")
        f.write(f"  {output_dir}/test/speaker_to_shard.json\n")
    
    print(f"\n✓ Dataset summary written to {summary_path}")
    
    print("\n" + "=" * 80)
    print("Dataset Creation Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print(f"\nTo load the dataset in your code:")
    print("  from datasets import load_dataset")
    print(f"  ")
    print(f"  # Load entire subset with streaming")
    print(f"  easy_dataset = load_dataset('parquet', data_dir='{output_dir}/easy', streaming=True)['train']")
    print(f"  hard_dataset = load_dataset('parquet', data_dir='{output_dir}/hard', streaming=True)['train']")
    print(f"  test_dataset = load_dataset('parquet', data_dir='{output_dir}/test', streaming=True)['train']")
    print(f"  ")
    print(f"  # Filter for specific speaker(s) - efficient with sharding")
    print(f"  speaker_dataset = easy_dataset.filter(lambda x: x['speaker_id'] == 'spk001')")
    print(f"  ")
    print(f"  # Use with PyTorch DataLoader (multiple workers supported)")
    print(f"  from torch.utils.data import DataLoader")
    print(f"  loader = DataLoader(easy_dataset, num_workers=4, batch_size=32)")
    print()
    
    return {
        'easy_speakers': len(easy_speakers),
        'hard_speakers': len(hard_speakers),
        'test_speakers': len(test_speakers),
        'easy_utterances': len(easy_subset),
        'hard_utterances': len(hard_subset),
        'test_utterances': len(test_samples)
    }

if __name__ == "__main__":
    results = create_parquet_dataset(
        titw_data_dir='../../training/data/titw',
        output_dir='../../training/data/titw_hf'
    )
    print(f"\nFinal statistics:")
    print(f"  Easy: {results['easy_speakers']} speakers, {results['easy_utterances']} utterances")
    print(f"  Hard: {results['hard_speakers']} speakers, {results['hard_utterances']} utterances")
    print(f"  Test: {results['test_speakers']} speakers, {results['test_utterances']} utterances")
