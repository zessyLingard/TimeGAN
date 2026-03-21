/* bch_decode.c
 * BCH Decoder with AES-256-CBC for IAT covert channel (LSB-first)
 * Usage: ./bch_decode <m> <length> <t> <timing_file> <threshold>
 *
 * timing_file may be CSV or newline-separated timings (ms). Threshold is in same units.
 * Pipeline: IAT input -> BCH decode -> AES-256-CBC decrypt (with PKCS7 unpadding) -> file
 * Reads AES key from aes_key.txt
 * 
 * Compile: gcc -o bch_decode bch_decode.c -lssl -lcrypto
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>

#define MAX_TIMINGS 10000000  /* grow if you need more */
#define AES_KEY_FILE "aes_key.txt"
#define AES_KEY_SIZE 32
#define AES_IV_SIZE 16

int m, n, length, k, t, d;
int p[21];
int alpha_to[1048576], index_of[1048576], g[548576];
int recd[1048576];

/* Load AES-256 key from hex file */
int load_aes_key(unsigned char *key) {
    FILE *fp = fopen(AES_KEY_FILE, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open key file '%s'\n", AES_KEY_FILE);
        return -1;
    }
    char hex[65];
    if (fscanf(fp, "%64s", hex) != 1 || strlen(hex) != 64) {
        fprintf(stderr, "Error: Key file must contain 64 hex characters\n");
        fclose(fp);
        return -1;
    }
    fclose(fp);
    
    for (int i = 0; i < 32; i++) {
        unsigned int byte;
        sscanf(hex + 2*i, "%2x", &byte);
        key[i] = (unsigned char)byte;
    }
    return 0;
}

/* AES-256-CBC decrypt (with PKCS7 unpadding) */
int aes_cbc_decrypt(const unsigned char *ciphertext, int ciphertext_len,
                    const unsigned char *key, const unsigned char *iv,
                    unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    
    int len, plaintext_len;
    
    /* Initialize AES-256-CBC decryption - padding removal is enabled by default */
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    
    /* Decrypt the ciphertext */
    if (EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len = len;
    
    /* Finalize decryption - this removes PKCS7 padding */
    if (EVP_DecryptFinal_ex(ctx, plaintext + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    return plaintext_len;
}

void generate_gf() {
    register int i, mask;
    
    mask = 1;
    alpha_to[m] = 0;
    for (i = 0; i < m; i++) {
        alpha_to[i] = mask;
        index_of[alpha_to[i]] = i;
        if (p[i] != 0)
            alpha_to[m] ^= mask;
        mask <<= 1;
    }
    index_of[alpha_to[m]] = m;
    mask >>= 1;
    for (i = m + 1; i < n; i++) {
        if (alpha_to[i - 1] >= mask)
            alpha_to[i] = alpha_to[m] ^ ((alpha_to[i - 1] ^ mask) << 1);
        else
            alpha_to[i] = alpha_to[i - 1] << 1;
        index_of[alpha_to[i]] = i;
    }
    index_of[0] = -1;
}

void gen_poly() {
    register int ii, jj, ll, kaux;
    register int test, aux, nocycles, root, noterms, rdncy;
    int cycle[1024][21], size[1024], min[1024], zeros[1024];
    
    cycle[0][0] = 0;
    size[0] = 1;
    cycle[1][0] = 1;
    size[1] = 1;
    jj = 1;
    
    do {
        ii = 0;
        do {
            ii++;
            cycle[jj][ii] = (cycle[jj][ii - 1] * 2) % n;
            size[jj]++;
            aux = (cycle[jj][ii] * 2) % n;
        } while (aux != cycle[jj][0]);
        
        ll = 0;
        do {
            ll++;
            test = 0;
            for (ii = 1; ((ii <= jj) && (!test)); ii++)
                for (kaux = 0; ((kaux < size[ii]) && (!test)); kaux++)
                    if (ll == cycle[ii][kaux])
                        test = 1;
        } while ((test) && (ll < (n - 1)));
        
        if (!(test)) {
            jj++;
            cycle[jj][0] = ll;
            size[jj] = 1;
        }
    } while (ll < (n - 1));
    
    nocycles = jj;
    d = 2 * t + 1;
    
    kaux = 0;
    rdncy = 0;
    for (ii = 1; ii <= nocycles; ii++) {
        min[kaux] = 0;
        test = 0;
        for (jj = 0; ((jj < size[ii]) && (!test)); jj++)
            for (root = 1; ((root < d) && (!test)); root++)
                if (root == cycle[ii][jj]) {
                    test = 1;
                    min[kaux] = ii;
                }
        if (min[kaux]) {
            rdncy += size[min[kaux]];
            kaux++;
        }
    }
    
    noterms = kaux;
    kaux = 1;
    for (ii = 0; ii < noterms; ii++)
        for (jj = 0; jj < size[min[ii]]; jj++) {
            zeros[kaux] = cycle[min[ii]][jj];
            kaux++;
        }
    
    k = length - rdncy;
    
    if (k < 0) {
        fprintf(stderr, "Error: Invalid BCH parameters!\n");
        exit(1);
    }
    
    g[0] = alpha_to[zeros[1]];
    g[1] = 1;
    for (ii = 2; ii <= rdncy; ii++) {
        g[ii] = 1;
        for (jj = ii - 1; jj > 0; jj--)
            if (g[jj] != 0)
                g[jj] = g[jj - 1] ^ alpha_to[(index_of[g[jj]] + zeros[ii]) % n];
            else
                g[jj] = g[jj - 1];
        g[0] = alpha_to[(index_of[g[0]] + zeros[ii]) % n];
    }
}

/* decode_bch() - unchanged (classic Berlekamp implementation) */
int decode_bch() {
    register int i, j, u, q, t2, count = 0, syn_error = 0;
    int elp[1026][1024], d_array[1026], l[1026], u_lu[1026], s[1025];
    int root[200], loc[200], reg[201];
    
    t2 = 2 * t;
    
    /* Form syndromes */
    for (i = 1; i <= t2; i++) {
        s[i] = 0;
        for (j = 0; j < length; j++)
            if (recd[j] != 0)
                s[i] ^= alpha_to[(i * j) % n];
        if (s[i] != 0)
            syn_error = 1;
        s[i] = index_of[s[i]];
    }
    
    if (syn_error) {
        /* Berlekamp algorithm */
        d_array[0] = 0;
        d_array[1] = s[1];
        elp[0][0] = 0;
        elp[1][0] = 1;
        for (i = 1; i < t2; i++) {
            elp[0][i] = -1;
            elp[1][i] = 0;
        }
        l[0] = 0;
        l[1] = 0;
        u_lu[0] = -1;
        u_lu[1] = 0;
        u = 0;
        
        do {
            u++;
            if (d_array[u] == -1) {
                l[u + 1] = l[u];
                for (i = 0; i <= l[u]; i++) {
                    elp[u + 1][i] = elp[u][i];
                    elp[u][i] = index_of[elp[u][i]];
                }
            } else {
                q = u - 1;
                while ((d_array[q] == -1) && (q > 0))
                    q--;
                
                if (q > 0) {
                    j = q;
                    do {
                        j--;
                        if ((d_array[j] != -1) && (u_lu[q] < u_lu[j]))
                            q = j;
                    } while (j > 0);
                }
                
                if (l[u] > l[q] + u - q)
                    l[u + 1] = l[u];
                else
                    l[u + 1] = l[q] + u - q;
                
                for (i = 0; i < t2; i++)
                    elp[u + 1][i] = 0;
                for (i = 0; i <= l[q]; i++)
                    if (elp[q][i] != -1)
                        elp[u + 1][i + u - q] = 
                            alpha_to[(d_array[u] + n - d_array[q] + elp[q][i]) % n];
                for (i = 0; i <= l[u]; i++) {
                    elp[u + 1][i] ^= elp[u][i];
                    elp[u][i] = index_of[elp[u][i]];
                }
            }
            u_lu[u + 1] = u - l[u + 1];
            
            if (u < t2) {
                if (s[u + 1] != -1)
                    d_array[u + 1] = alpha_to[s[u + 1]];
                else
                    d_array[u + 1] = 0;
                for (i = 1; i <= l[u + 1]; i++)
                    if ((s[u + 1 - i] != -1) && (elp[u + 1][i] != 0))
                        d_array[u + 1] ^= alpha_to[(s[u + 1 - i] + 
                                          index_of[elp[u + 1][i]]) % n];
                d_array[u + 1] = index_of[d_array[u + 1]];
            }
        } while ((u < t2) && (l[u + 1] <= t));
        
        u++;
        if (l[u] <= t) {
            /* Put elp into index form */
            for (i = 0; i <= l[u]; i++)
                elp[u][i] = index_of[elp[u][i]];
            
            /* Chien search */
            for (i = 1; i <= l[u]; i++)
                reg[i] = elp[u][i];
            count = 0;
            for (i = 1; i <= n; i++) {
                q = 1;
                for (j = 1; j <= l[u]; j++)
                    if (reg[j] != -1) {
                        reg[j] = (reg[j] + j) % n;
                        q ^= alpha_to[reg[j]];
                    }
                if (!q) {
                    root[count] = i;
                    loc[count] = n - i;
                    count++;
                }
            }
            
            if (count == l[u])
                for (i = 0; i < l[u]; i++)
                    recd[loc[i]] ^= 1;
            else
                return -1;  /* Uncorrectable error */
        } else {
            return -1;  /* Too many errors */
        }
    }
    
    return count;  /* Number of corrected errors */
}

/* Initialize bch (same as encoder) */
int bch_init(int code_m, int code_length, int code_t) {
    int i;
    
    m = code_m;
    length = code_length;
    t = code_t;
    n = (1 << m) - 1;
    
    for (i = 1; i < m; i++)
        p[i] = 0;
    p[0] = p[m] = 1;
    
    if (m == 8) p[4] = p[5] = p[6] = 1;
    else if (m == 7) p[1] = 1;
    else if (m == 6) p[1] = 1;
    else if (m == 5) p[2] = 1;
    else if (m == 4) p[1] = 1;
    else {
        fprintf(stderr, "Error: Unsupported m value: %d\n", m);
        return -1;
    }
    
    generate_gf();
    gen_poly();
    
    return k;
}

/* Parse timing file (CSV/newline tolerant), produce array of timings */
double *read_timings(const char *fname, long *out_count) {
    FILE *fp = fopen(fname, "r");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc(fsize + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t r = fread(buf,1,fsize,fp);
    buf[r] = '\0';
    fclose(fp);

    double *timings = malloc(sizeof(double) *  (MAX_TIMINGS));
    if (!timings) { free(buf); return NULL; }
    long count = 0;

    /* Tokenize by commas and whitespace */
    char *saveptr = NULL;
    char *tok = strtok_r(buf, ",\r\n\t ", &saveptr);
    while (tok && count < MAX_TIMINGS) {
        /* Skip empty tokens */
        if (*tok != '\0') {
            timings[count++] = atof(tok);
        }
        tok = strtok_r(NULL, ",\r\n\t ", &saveptr);
    }

    free(buf);
    *out_count = count;
    return timings;
}

/* Convert a block of timing values into recd[] (LSB-first).
 * timings: pointer to timings for the block (length entries).
 * threshold: cut to decide 1 vs 0 (timing <= threshold -> 1).
 */
int timings_to_recd(double *timings, int length, double threshold) {
    /* reconstructed_bytes holds bits LSB-first per byte */
    int bytes = (length + 7) / 8;
    unsigned char *reconstructed_bytes = calloc(bytes, 1);
    if (!reconstructed_bytes) return -1;

    for (int bit_count = 0; bit_count < length; bit_count++) {
        double timing = timings[bit_count];
        int bit_value = (timing <= threshold) ? 1 : 0;
        int byte_idx = bit_count / 8;
        int bit_pos = bit_count % 8; /* LSB-first */
        if (bit_value) reconstructed_bytes[byte_idx] |= (1 << bit_pos);
    }

    /* Fill recd[] in LSB-first order */
    memset(recd, 0, sizeof(int)* (length + 1));
    for (int i = 0; i < length; i++) {
        int byte_idx = i / 8;
        int bit_pos = i % 8;
        recd[i] = (reconstructed_bytes[byte_idx] >> bit_pos) & 1;
    }

    free(reconstructed_bytes);
    return 0;
}

/* Extract data bits from recd[] (last k bits) into output_data (LSB-first) */
/* Only extract k/8*8 bits (184 bits = 23 bytes) to match encoder */
void extract_data_bits_from_recd(unsigned char *output_data, int bytes_to_extract) {
    memset(output_data, 0, bytes_to_extract);
    int bits_to_extract = bytes_to_extract * 8;
    for (int i = 0; i < bits_to_extract && i < k; i++) {
        int codeword_pos = i + (length - k);
        if (recd[codeword_pos]) {
            int byte_idx = i / 8;
            int bit_pos = i % 8; /* LSB-first */
            output_data[byte_idx] |= (1 << bit_pos);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        printf("Usage: %s <m> <length> <t> <timing_file> <threshold>\n", argv[0]);
        printf("Example: %s 8 255 8 timing_values.txt 15.0\n", argv[0]);
        printf("BCH decoder with AES-256-CBC for covert timing channels\n");
        printf("Reads AES key from %s\n", AES_KEY_FILE);
        return 1;
    }

    int code_m = atoi(argv[1]);
    int code_length = atoi(argv[2]);
    int code_t = atoi(argv[3]);
    char *timing_file = argv[4];
    double threshold = atof(argv[5]);

    fprintf(stderr, "=== BCH Decoder with AES-256-CBC (LSB-first) ===\n");
    fprintf(stderr, "[1] Params: m=%d length=%d t=%d threshold=%.3f\n", code_m, code_length, code_t, threshold);

    /* Load AES key */
    unsigned char aes_key[AES_KEY_SIZE];
    if (load_aes_key(aes_key) != 0) {
        return 1;
    }
    fprintf(stderr, "[2] AES key loaded from %s\n", AES_KEY_FILE);

    int actual_k = bch_init(code_m, code_length, code_t);
    if (actual_k < 0) {
        fprintf(stderr, "Failed to initialize BCH\n");
        return 1;
    }
    fprintf(stderr, "[3] BCH initialized: length=%d data_bits=%d (k)\n", length, actual_k);

    long timing_count = 0;
    double *timings = read_timings(timing_file, &timing_count);
    if (!timings) {
        fprintf(stderr, "Error reading timing file '%s'\n", timing_file);
        return 1;
    }
    fprintf(stderr, "[4] Read %ld timing values from %s\n", timing_count, timing_file);

    /* Process each full codeword (length timings) */
    int blocks = timing_count / length;
    if (blocks == 0) {
        fprintf(stderr, "Warning: no full codeword blocks present (need %d timings per block)\n", length);
    }
    
    /* Collect all decoded bytes (will be nonce + ciphertext) */
    int bytes_per_block = k / 8;  /* Use floor: 191/8=23 bytes to match encoder */
    long max_decoded_bytes = (long)blocks * bytes_per_block;
    unsigned char *all_decoded = malloc(max_decoded_bytes);
    if (!all_decoded) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(timings);
        return 1;
    }
    long total_decoded_bytes = 0;
    int total_errors = 0;
    int failed_blocks = 0;
    
    for (int b = 0; b < blocks; b++) {
        double *block_timings = timings + (long)b * length;

        /* Convert timings to recd[] */
        if (timings_to_recd(block_timings, length, threshold) != 0) {
            fprintf(stderr, "Error reconstructing bits for block %d\n", b);
            free(timings);
            free(all_decoded);
            return 1;
        }

        /* Debug: show first 32 bits of recd[] */
        if (b == 0) {
            fprintf(stderr, "Block %d: first 32 received bits: ", b);
            for (int i = 0; i < 32 && i < length; i++) fprintf(stderr, "%d", recd[i]);
            fprintf(stderr, "\n");
        }

        /* Try BCH decode */
        int errors = decode_bch();
        if (errors < 0) {
            fprintf(stderr, "Block %d: BCH decode failed (uncorrectable)\n", b);
            failed_blocks++;
            /* Fill with zeros and continue */
            memset(all_decoded + total_decoded_bytes, 0, bytes_per_block);
            total_decoded_bytes += bytes_per_block;
            continue;
        }
        total_errors += errors;

        /* Extract k/8 data bytes and append to all_decoded */
        unsigned char out_bytes[bytes_per_block];
        extract_data_bits_from_recd(out_bytes, bytes_per_block);
        memcpy(all_decoded + total_decoded_bytes, out_bytes, bytes_per_block);
        total_decoded_bytes += bytes_per_block;
    }
    
    fprintf(stderr, "[5] BCH decoded: %ld bytes, %d errors corrected", total_decoded_bytes, total_errors);
    if (failed_blocks > 0) {
        fprintf(stderr, ", %d blocks failed", failed_blocks);
    }
    fprintf(stderr, "\n");

    free(timings);
    
    /* Check if we have at least IV + some data */
    if (total_decoded_bytes < AES_IV_SIZE) {
        fprintf(stderr, "Error: Decoded data too short (need at least %d bytes for IV)\n", AES_IV_SIZE);
        free(all_decoded);
        return 1;
    }
    
    /* Extract IV and ciphertext */
    unsigned char *iv = all_decoded;
    unsigned char *ciphertext = all_decoded + AES_IV_SIZE;
    long ciphertext_len = total_decoded_bytes - AES_IV_SIZE;
    
    /* AES-256-CBC decrypt */
    unsigned char *plaintext = malloc(ciphertext_len);
    if (!plaintext) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(all_decoded);
        return 1;
    }
    
    int plaintext_len = aes_cbc_decrypt(ciphertext, ciphertext_len, aes_key, iv, plaintext);
    if (plaintext_len < 0) {
        fprintf(stderr, "Error: AES-CBC decryption failed (padding error or wrong key)\n");
        free(all_decoded);
        free(plaintext);
        return 1;
    }
    fprintf(stderr, "[6] AES-CBC decrypted: %d bytes\n", plaintext_len);
    
    /* Write plaintext to stdout */
    fwrite(plaintext, 1, plaintext_len, stdout);
    fflush(stdout);
    
    fprintf(stderr, "\n=== Decoding complete ===\n");
    fprintf(stderr, "- Timing values: %ld\n", timing_count);
    fprintf(stderr, "- BCH blocks: %d\n", blocks);
    fprintf(stderr, "- Errors corrected: %d\n", total_errors);
    fprintf(stderr, "- Output: %d bytes\n", plaintext_len);

    free(all_decoded);
    free(plaintext);
    return 0;
}

