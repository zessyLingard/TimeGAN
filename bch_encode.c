/*
 * BCH Encoder for CCEAP Covert Timing Channel
 * Usage: ./bch_encode <input_file> <low_delay_ms> <high_delay_ms> [-k password]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_AES
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

#define BCH_M 8
#define BCH_N 255
#define BCH_T 8

int m, n, length, k, t, d;
int p[21];
int alpha_to[1048576], index_of[1048576], g[1048576];
int data[1048576], bb[1048576];

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

#ifdef USE_AES
/* AES-256-CTR encryption context */
static EVP_CIPHER_CTX *aes_ctx = NULL;
static unsigned char aes_key[32];
static unsigned char aes_iv[16];

/* Derive key from password using PBKDF2 */
int aes_init_encrypt(const char *password) {
    /* Fixed salt - in production, use random salt and transmit it */
    unsigned char salt[16] = "BCH_AES_SALT_01";
    
    /* Derive 32-byte key from password */
    if (PKCS5_PBKDF2_HMAC(password, strlen(password), salt, 16, 100000,
                          EVP_sha256(), 32, aes_key) != 1) {
        fprintf(stderr, "Error: Key derivation failed\n");
        return -1;
    }
    
    /* Use fixed IV derived from password (for reproducibility) */
    if (PKCS5_PBKDF2_HMAC(password, strlen(password), salt, 16, 100000,
                          EVP_sha256(), 16, aes_iv) != 1) {
        fprintf(stderr, "Error: IV derivation failed\n");
        return -1;
    }
    
    aes_ctx = EVP_CIPHER_CTX_new();
    if (!aes_ctx) return -1;
    
    if (EVP_EncryptInit_ex(aes_ctx, EVP_aes_256_ctr(), NULL, aes_key, aes_iv) != 1) {
        EVP_CIPHER_CTX_free(aes_ctx);
        return -1;
    }
    
    fprintf(stderr, "AES-256-CTR encryption initialized\n");
    return 0;
}

/* Encrypt data in-place using AES-CTR (XOR with keystream) */
int aes_encrypt_block(unsigned char *data, int len) {
    if (!aes_ctx) return len;  /* No encryption */
    
    unsigned char outbuf[64];
    int outlen;
    
    if (EVP_EncryptUpdate(aes_ctx, outbuf, &outlen, data, len) != 1) {
        return -1;
    }
    memcpy(data, outbuf, outlen);
    return outlen;
}

void aes_cleanup() {
    if (aes_ctx) {
        EVP_CIPHER_CTX_free(aes_ctx);
        aes_ctx = NULL;
    }
}

/* Read password from pass.txt file */
char* read_password_file() {
    static char password[256];
    FILE *fp = fopen("pass.txt", "r");
    if (!fp) {
        fprintf(stderr, "Warning: pass.txt not found, encryption disabled\n");
        return NULL;
    }
    if (fgets(password, sizeof(password), fp) == NULL) {
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    /* Remove trailing newline */
    size_t len = strlen(password);
    while (len > 0 && (password[len-1] == '\n' || password[len-1] == '\r')) {
        password[--len] = '\0';
    }
    if (len == 0) return NULL;
    return password;
}
#endif

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
        fprintf(stderr, "BCH(%d,k,%d) encoder for covert timing channels\n", BCH_N, 2*BCH_T+1);
        fprintf(stderr, "  AES-256-CTR encryption: place password in pass.txt\n");
        return 1;
    }
    
    char* input_file = argv[1];
    double low_delay = atof(argv[2]);
    double high_delay = atof(argv[3]);
    char* password = NULL;
    
    if (low_delay <= 0 || high_delay <= 0 || low_delay >= high_delay) {
        fprintf(stderr, "Error: Invalid delay values\n");
        return 1;
    }
    
#ifdef USE_AES
    /* Auto-read password from pass.txt */
    password = read_password_file();
    if (password) {
        if (aes_init_encrypt(password) != 0) {
            fprintf(stderr, "Error: Failed to initialize AES encryption\n");
            return 1;
        }
    }
#endif
    
    int actual_k = bch_init();
    if (actual_k <= 0) {
        fprintf(stderr, "Error: Failed to initialize BCH code\n");
        return 1;
    }
    
    
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
    
    int data_bytes_per_block = (actual_k + 7) / 8;
    int total_blocks = (file_size + data_bytes_per_block - 1) / data_bytes_per_block;
    int total_bits_encoded = 0;
    int first_output = 1;
    
    fprintf(stderr, "Processing %ld bytes in %d blocks\n", file_size, total_blocks);
    fprintf(stderr, "IMPORTANT: Original file size = %ld bytes (pass to decoder with -s flag)\n", file_size);
    
    for (int block_num = 0; block_num < total_blocks; block_num++) {
        unsigned char block_data[32];
        memset(block_data, 0, sizeof(block_data));
        
        int bytes_read = fread(block_data, 1, data_bytes_per_block, fp);
        if (bytes_read <= 0) break;
        
#ifdef USE_AES
        /* Encrypt block before BCH encoding */
        if (password) {
            aes_encrypt_block(block_data, bytes_read);
        }
#endif
        
        // Convert bytes to bits and store in data array
        memset(data, 0, sizeof(data));
        for (int i = 0; i < bytes_read * 8 && i < actual_k; i++) {
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
    
    fclose(fp);

    fprintf(stderr, "\nEncoding complete:\n");
    fprintf(stderr, "- Input: %ld bytes\n", file_size);
    fprintf(stderr, "- Blocks: %d\n", total_blocks);
    fprintf(stderr, "- Encoded bits: %d\n", total_bits_encoded);
    fprintf(stderr, "- Total IAT values: %d\n", total_bits_encoded);
    fprintf(stderr, "- Overall rate: %.4f\n", (float)(file_size * 8) / total_bits_encoded);
    if (password) fprintf(stderr, "- Encryption: AES-256-CTR\n");
    
#ifdef USE_AES
    aes_cleanup();
#endif
    return 0;
}

