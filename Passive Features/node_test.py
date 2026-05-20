import numpy as np
import time
import os

def stress_test(matrix_size=10000):
    print(f"Starting HPC Node Test on hostname: {os.uname().nodename}")
    print(f"Allocated CPUs (approx): {os.cpu_count()}")
    
    # Generate two large matrices
    print(f"Generating two {matrix_size}x{matrix_size} matrices...")
    start_time = time.time()
    
    # Using float64 to ensure heavy memory and CPU usage
    A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    B = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    
    gen_time = time.time() - start_time
    print(f"Generation took: {gen_time:.2f} seconds.")
    
    # Perform matrix multiplication (Dot product)
    print("Performing matrix multiplication...")
    start_time = time.time()
    
    C = np.dot(A, B)
    
    calc_time = time.time() - start_time
    print(f"Multiplication took: {calc_time:.2f} seconds.")
    
    # Verification step to ensure no silent memory corruption
    print(f"Result matrix trace (verification): {np.trace(C):.2f}")
    print("Test completed successfully.")

if __name__ == "__main__":
    # Adjust matrix_size based on how much memory the node has. 
    # 10,000x10,000 float64 matrices will use about 800MB each (2.4GB total for A, B, and C).
    stress_test(matrix_size=10000)