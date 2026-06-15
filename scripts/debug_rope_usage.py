"""Debug script to verify RoPE usage in SGLang before kunpeng integration.

This script adds debug prints to understand:
1. positions shape and dtype
2. q_pe / k_pe shapes
3. rotary_emb cache format and shape
4. whether offsets are used

Run:
    python scripts/debug_rope_usage.py
"""

import sys
import os

# Patch forward_mla.py to add debug prints
def add_debug_prints():
    forward_mla_path = "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py"

    if not os.path.exists(forward_mla_path):
        print(f"Error: {forward_mla_path} not found")
        return False

    with open(forward_mla_path, 'r') as f:
        content = f.read()

    # Find the line: q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    if "q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)" not in content:
        print("Error: Could not find rotary_emb call in forward_mla.py")
        return False

    # Add debug prints before the call
    debug_code = '''
        # DEBUG: Check RoPE inputs
        print(f"\\n=== RoPE Debug Info (Layer {self.layer_id}) ===")
        print(f"positions shape: {positions.shape}, dtype: {positions.dtype}")
        print(f"positions sample: {positions[:min(5, positions.shape[0])]}")
        print(f"q_pe shape: {q_pe.shape}, dtype: {q_pe.dtype}")
        print(f"k_pe shape: {k_pe.shape}, dtype: {k_pe.dtype}")
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            print(f"rotary_emb type: {type(self.rotary_emb).__name__}")
            if hasattr(self.rotary_emb, 'long_short_cos_sin_cache'):
                cache = self.rotary_emb.long_short_cos_sin_cache
                print(f"cache shape: {cache.shape}, dtype: {cache.dtype}")
                # Check cache format: [cos..., sin...] or [cos, sin, cos, sin, ...]
                mid = cache.shape[1] // 2
                cos_part = cache[0, :mid]
                sin_part = cache[0, mid:]
                print(f"cache[0, :mid] (cos part) sample: {cos_part[:5]}")
                print(f"cache[0, mid:] (sin part) sample: {sin_part[:5]}")
            elif hasattr(self.rotary_emb, 'cos_sin_cache'):
                cache = self.rotary_emb.cos_sin_cache
                print(f"cos_sin_cache shape: {cache.shape}, dtype: {cache.dtype}")
        print("=== End RoPE Debug ===\\n")
        '''

    # Insert before the rotary_emb call
    modified_content = content.replace(
        "            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)",
        debug_code + "            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)"
    )

    # Write back
    with open(forward_mla_path, 'w') as f:
        f.write(modified_content)

    print(f"✅ Added debug prints to {forward_mla_path}")
    return True


def check_rope_variant():
    """Check if RotaryEmbedding.forward has offsets parameter."""
    rope_path = "python/sglang/srt/layers/rotary_embedding/rope_variant.py"

    with open(rope_path, 'r') as f:
        content = f.read()

    # Find RotaryEmbedding.forward signature
    import re
    match = re.search(r'class RotaryEmbedding.*?def forward\(.*?\):', content, re.DOTALL)
    if match:
        signature = match.group(0)
        print("\n=== RotaryEmbedding.forward signature ===")
        print(signature)

        has_offsets = 'offsets' in signature
        print(f"\nHas 'offsets' parameter: {has_offsets}")

        return has_offsets

    return None


def main():
    print("=" * 60)
    print("RoPE Usage Debug Script")
    print("=" * 60)

    # Step 1: Check rope_variant.py
    print("\nStep 1: Checking RotaryEmbedding signature...")
    check_rope_variant()

    # Step 2: Add debug prints
    print("\nStep 2: Adding debug prints to forward_mla.py...")
    success = add_debug_prints()

    if not success:
        print("\n❌ Failed to add debug prints")
        return

    print("\n" + "=" * 60)
    print("✅ Debug prints added successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run a simple inference:")
    print("   python -m sglang.launch_server \\")
    print("     --model-path meituan/DeepSeek-R1-Channel-INT8 \\")
    print("     --quantization w8a8_int8 \\")
    print("     --port 30000")
    print("")
    print("2. Send a test request:")
    print("   curl http://localhost:30000/generate \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"text\": \"Hello\", \"sampling_params\": {\"max_new_tokens\": 8}}'")
    print("")
    print("3. Check the server logs for debug output")
    print("4. Send me the debug output, especially:")
    print("   - positions shape (1D or 2D?)")
    print("   - cache format (cos/sin layout)")
    print("   - whether offsets are used")
    print("")
    print("⚠️  Remember to revert the debug prints after verification:")
    print("   git checkout python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
