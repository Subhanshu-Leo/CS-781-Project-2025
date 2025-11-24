"""
Debug script to understand VerifAI sampler output structure
Run this to see what format the samples are in
"""

from verifai.features import Struct, Box, Feature, FeatureSpace
from verifai.samplers import FeatureSampler
import numpy as np

# Define feature space
control_params = Struct({
    'lateral_offset': Box([-0.5, 0.5]),
    'lateral_velocity': Box([-0.1, 0.1]),
    'heading_error': Box([-0.175, 0.175]),
    'heading_rate': Box([-0.05, 0.05])
})

sample_space = FeatureSpace({
    'params': Feature(control_params)
})

# Create sampler
sampler = FeatureSampler.randomSamplerFor(sample_space)

# Get a sample and inspect its structure
print("\n" + "="*70)
print("DEBUGGING VERIFAI SAMPLER OUTPUT STRUCTURE")
print("="*70)

for i in range(3):
    sample = sampler.nextSample()
    print(f"\nSample {i+1}:")
    print(f"  Type: {type(sample)}")
    print(f"  Value: {sample}")
    print(f"  Dir: {[attr for attr in dir(sample) if not attr.startswith('_')]}")

    # Try different access methods
    print(f"\n  Access attempts:")

    # Method 1: Direct attribute
    try:
        print(f"    sample.params: {sample.params}")
        print(f"    sample.params type: {type(sample.params)}")
        if hasattr(sample.params, '__iter__'):
            print(f"    sample.params[0]: {sample.params[0]}")
    except Exception as e:
        print(f"    sample.params failed: {e}")

    # Method 2: Dictionary access
    try:
        print(f"    sample['params']: {sample['params']}")
    except Exception as e:
        print(f"    sample['params'] failed: {e}")

    # Method 3: Iterate and print all attributes
    try:
        if hasattr(sample, '__dict__'):
            print(f"    sample.__dict__: {sample.__dict__}")
    except Exception as e:
        print(f"    sample.__dict__ failed: {e}")

    # Method 4: Check if it's tuple-like
    try:
        print(f"    As tuple: {tuple(sample)}")
    except Exception as e:
        print(f"    As tuple failed: {e}")

print("\n" + "="*70)
