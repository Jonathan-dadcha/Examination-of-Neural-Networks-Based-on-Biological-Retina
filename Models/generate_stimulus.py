import os
import numpy as np
import h5py
from config import BASE_PATH, SESSION

# --- C++ SOURCE CODE ---
cpp_code = """
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

// Standard Numerical Recipes ran1 algorithm
float ran1(long *idum) {
    int j;
    long k;
    static long iy=0;
    static long iv[NTAB];
    float temp;

    if (*idum <= 0 || !iy) {
        if (-(*idum) < 1) *idum=1;
        else *idum = -(*idum);
        for (j=NTAB+7;j>=0;j--) {
            k=(*idum)/IQ;
            *idum=IA*(*idum-k*IQ)-IR*k;
            if (*idum < 0) *idum += IM;
            if (j < NTAB) iv[j] = *idum;
        }
        iy=iv[0];
    }
    k=(*idum)/IQ;
    *idum=IA*(*idum-k*IQ)-IR*k;
    if (*idum < 0) *idum += IM;
    j=iy/NDIV;
    iy=iv[j];
    iv[j] = *idum;
    if ((temp=AM*iy) > RNMX) return RNMX;
    else return temp;
}

int main() {
    long seed = -10000;  // Fixed seed for session 20171116_sr_le_fp
    int Nx = 75;         // Width
    int Ny = 100;        // Height
    int frames = 75000;  // Full duration
    
    // --- VERSION 1: Loop Order (Time -> Y -> X) ---
    // Matches standard image processing (Row-Major)
    long idum1 = seed;
    ran1(&idum1); // Init
    
    std::ofstream out1("wn_v1.bin", std::ios::binary);
    if (!out1) return 1;
    
    for (int t=0; t<frames; t++) {
        for (int y=0; y<Ny; y++) {
            for (int x=0; x<Nx; x++) {
                float r = ran1(&idum1);
                uint8_t val = (r > 0.5) ? 255 : 0;
                out1.write((char*)&val, sizeof(uint8_t));
            }
        }
    }
    out1.close();
    
    // --- VERSION 2: Loop Order (Time -> X -> Y) ---
    // Matches MATLAB column-major linear indexing
    long idum2 = seed;
    ran1(&idum2); // Init (Reset sequence)
    
    std::ofstream out2("wn_v2.bin", std::ios::binary);
    if (!out2) return 1;

    for (int t=0; t<frames; t++) {
        for (int x=0; x<Nx; x++) {      // SWAPPED LOOP
            for (int y=0; y<Ny; y++) {  // SWAPPED LOOP
                float r = ran1(&idum2);
                uint8_t val = (r > 0.5) ? 255 : 0;
                out2.write((char*)&val, sizeof(uint8_t));
            }
        }
    }
    out2.close();
    
    return 0;
}
"""

def generate_and_convert():
    # 1. Write C++ file
    with open("generator.cpp", "w") as f:
        f.write(cpp_code)
    
    print("🔨 Compiling C++ generator...")
    ret = os.system("c++ generator.cpp -o generator -O3")
    if ret != 0:
        print("❌ Compilation failed.")
        return
    
    print("🚀 Running C++ generator (creating 2 versions)...")
    ret = os.system("./generator")
    if ret != 0:
        print("❌ Execution failed.")
        return
    
    # 2. Convert Binary to H5
    _output_dir = os.path.join(BASE_PATH, SESSION, 'processed_data')
    os.makedirs(_output_dir) if not os.path.exists(_output_dir) else None

    # Convert Version 1
    print("💾 Converting Version 1 to HDF5...")
    with open("wn_v1.bin", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # Shape: (Frames, Height, Width)
    data = data.reshape((75000, 100, 75))
    
    with h5py.File(os.path.join(_output_dir, 'white_noise_v1.h5'), 'w') as f:
        f.create_dataset('stimulus', data=data)
        
    # Convert Version 2 (Transpose needed to match layout)
    print("💾 Converting Version 2 to HDF5...")
    with open("wn_v2.bin", "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # Generated as (Time, X, Y), reshape accordingly
    data = data.reshape((75000, 75, 100))
    # Transpose to (Time, Height, Width) = (Time, Y, X)
    data = np.transpose(data, (0, 2, 1))
    
    with h5py.File(os.path.join(_output_dir, 'white_noise_v2.h5'), 'w') as f:
        f.create_dataset('stimulus', data=data)
        
    print("\n✅ SUCCESS! Two versions generated:")
    print(f"   1. {os.path.join(_output_dir, 'white_noise_v1.h5')}")
    print(f"   2. {os.path.join(_output_dir, 'white_noise_v2.h5')}")
    print("   (Temporary files cleaned up)")
    
    os.remove("generator.cpp")
    os.remove("generator")
    os.remove("wn_v1.bin")
    os.remove("wn_v2.bin")

if __name__ == "__main__":
    generate_and_convert()