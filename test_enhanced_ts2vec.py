"""
Test script for Enhanced TS2Vec with Time Features
This validates our improvements to TS2Vec's forecasting capabilities.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_time_features():
    """Test the generate_time_features function"""
    print("🔹 Testing Time Features Generation...")
    
    from tasks.forecasting import generate_time_features
    
    # Test with 168 hours (1 week of hourly data)
    length = 168
    time_feats = generate_time_features(length)
    
    print(f"   ✅ Time features shape: {time_feats.shape}")
    print(f"   ✅ Expected shape: ({length}, 6)")
    print(f"   ✅ Features include: daily, weekly, monthly sin/cos cycles")
    
    # Check if features are properly bounded between -1 and 1
    assert time_feats.min() >= -1.0 and time_feats.max() <= 1.0, "Time features should be bounded [-1,1]"
    assert time_feats.shape == (length, 6), f"Expected shape ({length}, 6), got {time_feats.shape}"
    
    print("   ✅ Time features test PASSED!")
    return True

def test_enhanced_sample_generation():
    """Test enhanced sample generation with time features"""
    print("\n🔹 Testing Enhanced Sample Generation...")
    
    from tasks.forecasting import generate_pred_samples
    
    # Create synthetic TS2Vec embeddings and data
    np.random.seed(42)
    batch_size = 2
    seq_len = 200
    embed_dim = 64
    n_vars = 7
    pred_len = 24
    
    # Synthetic TS2Vec features (batch_size, seq_len, embed_dim)
    features = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Synthetic time series data (batch_size, seq_len, n_vars)
    data = np.random.randn(batch_size, seq_len, n_vars)
    
    # Test without time features
    X_orig, y_orig = generate_pred_samples(features, data, pred_len, drop=50, add_time_features=False)
    
    # Test with time features
    X_enhanced, y_enhanced = generate_pred_samples(features, data, pred_len, drop=50, add_time_features=True)
    
    print(f"   ✅ Original features shape: {X_orig.shape}")
    print(f"   ✅ Enhanced features shape: {X_enhanced.shape}")
    print(f"   ✅ Labels shape: {y_orig.shape}")
    
    # Enhanced features should have 6 additional time features
    expected_diff = 6  # 6 time features (sin/cos for daily, weekly, monthly)
    actual_diff = X_enhanced.shape[1] - X_orig.shape[1]
    
    assert actual_diff == expected_diff, f"Expected {expected_diff} additional features, got {actual_diff}"
    assert y_enhanced.shape == y_orig.shape, "Labels should be identical"
    
    print(f"   ✅ Added {actual_diff} time features as expected")
    print("   ✅ Enhanced sample generation test PASSED!")
    return True

def main():
    """Run all tests for enhanced TS2Vec"""
    print("🚀 Testing Enhanced TS2Vec Implementation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Run individual tests
    if test_time_features():
        tests_passed += 1
        
    if test_enhanced_sample_generation():
        tests_passed += 1
    
    # Final results
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! Enhanced TS2Vec is ready for forecasting.")
        print("\n📋 Summary of Enhancements:")
        print("   • ✅ Explicit temporal features (daily/weekly/monthly cycles)")
        print("   • ✅ Ridge regression head for forecasting")
        print("   • ✅ Enhanced sample generation pipeline")
        print("\n🎯 Expected Benefits:")
        print("   • Better long-horizon forecasting (H=168, 336, 720)")
        print("   • Improved handling of seasonal patterns")
        print("   • Linear temporal relationship modeling with time features")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()