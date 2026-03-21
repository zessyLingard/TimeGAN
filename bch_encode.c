/*
 * BCH Encoder with AES-256-CBC for CCEAP Covert Timing Channel
 * Usage: ./bch_encode <input_file> <low_delay_ms> <high_delay_ms>
 * 
 * Reads AES key from aes_key.txt (64 hex chars = 32 bytes)
 * Pipeline: file -> AES-256-CBC encrypt (with PKCS7 padding) -> BCH encode -> IAT output
 * 
 * Compile: gcc -o bch_encode bch_encode.c -lssl -lcrypto
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#define BCH_M 8
#define BCH_N 255
#define BCH_T 8
#define AES_KEY_FILE "aes_key.txt"
#define AES_KEY_SIZE 32
#define AES_IV_SIZE 16
#define AES_BLOCK_SIZE 16

int m, n, length, k, t, d;
int p[21];
int alpha_to[1048576], index_of[1048576], g[1048576];
int data[1048576], bb[1048576];

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

/* AES-256-CBC encrypt (with PKCS7 padding) */
int aes_cbc_encrypt(const unsigned char *plaintext, int plaintext_len,
                    const unsigned char *key, const unsigned char *iv,
                    unsigned char *ciphertext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    
    int len, ciphertext_len;
    
    /* Initialize AES-256-CBC encryption - padding is enabled by default */
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    
    /* Encrypt the plaintext */
    if (EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len = len;
    
    /* Finalize encryption - this adds PKCS7 padding */
    if (EVP_EncryptFinal_ex(ctx, ciphertext + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    return ciphertext_len;
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

void encode_bch() {
    register int i, j;
    register int feedback;
    
    for (i = 0; i < length - k; i++)
        bb[i] = 0;
    
    for (i = k - 1; i >= 0; i--) {
        feedback = data[i] ^ bb[length - k - 1];
        if (feedback != 0) {
            for (j = length - k - 1; j > 0; j--)
                if (g[j] != 0)
                    bb[j] = bb[j - 1] ^ feedback;
                else
                    bb[j] = bb[j - 1];
            bb[0] = g[0] & feedback;
        } else {
            for (j = length - k - 1; j > 0; j--)
                bb[j] = bb[j - 1];
            bb[0] = 0;
        }
    }
}

int bch_init() {
    int i;
    
    m = BCH_M;
    length = BCH_N;
    t = BCH_T;
    n = (1 << m) - 1;
    
    for (i = 1; i < m; i++)
        p[i] = 0;
    p[0] = p[m] = 1;
    p[4] = p[5] = p[6] = 1;
    
    generate_gf();
    gen_poly();
    
    return k;
}

void output_block_bits(double low_delay, double high_delay, int *first_output) {
    int i;
    int codeword[255];
    for (i = 0; i < length - k; i++) {
        codeword[i] = bb[i];
    }
    for (i = 0; i < k; i++) {
        codeword[length - k + i] = data[i];
    }

    // Traverse codeword in LSB-first order (per byte)
    int total_bytes = (length + 7) / 8;
    for (int byte_num = 0; byte_num < total_bytes; byte_num++) {
        for (int bit_num = 0; bit_num < 8; bit_num++) {
            int bit_index = byte_num * 8 + bit_num;
            if (bit_index >= length) break;

            double iat = codeword[bit_index] ? low_delay : high_delay;

            if (*first_output) {
                printf("%.1f", iat);
                *first_output = 0;
            } else {
                printf(",%.1f", iat);
            }
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <low_delay_ms> <high_delay_ms>\n", argv[0]);
        fprintf(stderr, "BCH(%d,k,%d) encoder with AES-256-CBC for covert timing channels\n", BCH_N, 2*BCH_T+1);
        fprintf(stderr, "Reads AES key from %s\n", AES_KEY_FILE);
        return 1;
    }
    
    char* input_file = argv[1];
    double low_delay = atof(argv[2]);
    double high_delay = atof(argv[3]);
    
    if (low_delay <= 0 || high_delay <= 0 || low_delay >= high_delay) {
        fprintf(stderr, "Error: Invalid delay values\n");
        return 1;
    }
    
    /* Load AES key */
    unsigned char aes_key[AES_KEY_SIZE];
    if (load_aes_key(aes_key) != 0) {
        return 1;
    }
    fprintf(stderr, "[1] AES key loaded from %s\n", AES_KEY_FILE);
    
    int actual_k = bch_init();
    if (actual_k <= 0) {
        fprintf(stderr, "Error: Failed to initialize BCH code\n");
        return 1;
    }
    fprintf(stderr, "[2] BCH(%d,%d) initialized, t=%d\n", BCH_N, actual_k, BCH_T);
    
    /* Read input file */
    FILE* fp = fopen(input_file, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
        return 1;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if (file_size == 0) {
        fprintf(stderr, "Error: Input file is empty\n");
        fclose(fp);
        return 1;
    }
    
    unsigned char *plaintext = malloc(file_size);
    if (!plaintext) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return 1;
    }
    fread(plaintext, 1, file_size, fp);
    fclose(fp);
    fprintf(stderr, "[3] Read %ld bytes from %s\n", file_size, input_file);
    
    /* AES-256-CBC encrypt */
    unsigned char iv[AES_IV_SIZE];
    if (RAND_bytes(iv, AES_IV_SIZE) != 1) {
        fprintf(stderr, "Error: Failed to generate random IV\n");
        free(plaintext);
        return 1;
    }
    
    /* CBC ciphertext = IV (16 bytes) + encrypted data (padded to block size) */
    /* Max ciphertext size: IV + plaintext + up to 16 bytes padding */
    long max_ciphertext_len = AES_IV_SIZE + file_size + AES_BLOCK_SIZE;
    unsigned char *ciphertext = malloc(max_ciphertext_len);
    if (!ciphertext) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(plaintext);
        return 1;
    }
    
    /* Prepend IV */
    memcpy(ciphertext, iv, AES_IV_SIZE);
    
    /* Encrypt with CBC (includes PKCS7 padding) */
    int encrypted_len = aes_cbc_encrypt(plaintext, file_size, aes_key, iv, 
                                        ciphertext + AES_IV_SIZE);
    if (encrypted_len < 0) {
        fprintf(stderr, "Error: AES-CBC encryption failed\n");
        free(plaintext);
        free(ciphertext);
        return 1;
    }
    long ciphertext_len = AES_IV_SIZE + encrypted_len;
    fprintf(stderr, "[4] AES-CBC encrypted: %ld bytes (IV + padded ciphertext)\n", ciphertext_len);
    
    free(plaintext);
    
    /* BCH encode the ciphertext */
    int data_bytes_per_block = (actual_k + 7) / 8;
    int total_blocks = (ciphertext_len + data_bytes_per_block - 1) / data_bytes_per_block;
    int total_bits_encoded = 0;
    int first_output = 1;
    
    fprintf(stderr, "[5] BCH encoding %ld bytes in %d blocks\n", ciphertext_len, total_blocks);
    
    for (int block_num = 0; block_num < total_blocks; block_num++) {
        unsigned char block_data[32];
        memset(block_data, 0, sizeof(block_data));
        
        long offset = (long)block_num * data_bytes_per_block;
        int bytes_to_read = data_bytes_per_block;
        if (offset + bytes_to_read > ciphertext_len) {
            bytes_to_read = ciphertext_len - offset;
        }
        memcpy(block_data, ciphertext + offset, bytes_to_read);
        
        // Convert bytes to bits and store in data array
        memset(data, 0, sizeof(data));
        for (int i = 0; i < bytes_to_read * 8 && i < actual_k; i++) {
            int byte_idx = i / 8;
            int bit_idx = i % 8;  // LSB first
            data[i] = (block_data[byte_idx] >> bit_idx) & 1;
        }       

        encode_bch();
        
        // Output encoded block as IAT values
        output_block_bits(low_delay, high_delay, &first_output);
        
        total_bits_encoded += length;
        
        if ((block_num + 1) % 10 == 0) {
            fprintf(stderr, "Encoded %d/%d blocks\n", block_num + 1, total_blocks);
        }
    }
    
    free(ciphertext);

    fprintf(stderr, "\n[6] Encoding complete:\n");
    fprintf(stderr, "- Input: %ld bytes\n", file_size);
    fprintf(stderr, "- Ciphertext: %ld bytes (with nonce)\n", ciphertext_len);
    fprintf(stderr, "- Blocks: %d\n", total_blocks);
    fprintf(stderr, "- Encoded bits: %d\n", total_bits_encoded);
    fprintf(stderr, "- Total IAT values: %d\n", total_bits_encoded);
    fprintf(stderr, "- Overall rate: %.4f\n", (float)(file_size * 8) / total_bits_encoded);
    
    return 0;
}