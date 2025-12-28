
# Path tracer optimizations

## **Algorithmic / Rendering Quality:**

**1. Russian Roulette**
   Terminate paths probabilistically based on their contribution to reduce computation on low-impact rays.

   Running `cornell_box()` at:
   ```
   cam.img_width = 100
   cam.samples_per_pixel = 100
   cam.max_depth = 50
   ```

   | Technique | Time       |
   |-------|----------------|
   | Plain Python | 5m 5s  |
   | Russian Roulette | 3m 53s |

   Speedup: ~23% faster with Russian Roulette

**2. Importance Sampling**

   - **Cosine-weighted hemisphere sampling** for diffuse surfaces.
   One small change and the render quality improves significantly. Now the rays are more likely to be scattered in directions that contribute more light, less samples required to create same quality image as uniform sampling.

   ```python
   class lambertian(material):
      def scatter(...) -> bool:
         scatter_direction = random_cosine_direction(rec.normal)
         ...
   ```

   **IMP: Couldn't benchmark due to very slow rendering times**


## **Restructuring Data:**

   **3. Array of Structures of Arrays (AoSoA)**

   - Restructure data to improve memory access patterns and cache efficiency. Optimizing this specifically for efficient GPU access.

   
      