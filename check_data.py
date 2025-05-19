import numpy as np

def load_and_print_pos_src(file_path="/home/kato/AVR/MeshRIR/train/ir_18.npy"):
    """
    Load and print the contents of a .npy file containing source positions.

    Parameters:
    file_path (str): Path to the .npy file.
    """
    try:
        pos_src = np.load(file_path)
        print("Shape:", pos_src.shape)
        print("Data:\n", pos_src)
        return pos_src
    except Exception as e:
        print("Error loading file:", e)
        return None

# ===== 実行部分 =====
if __name__ == "__main__":
    load_and_print_pos_src()
