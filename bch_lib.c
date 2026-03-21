/*
 * Pure BCH Library - Bits In, Bits Out
 * Extracted from bch_encode.c and bch_decode.c
 * 
 * API:
 *   int bch_init(int m, int n, int t) - Initialize BCH code, returns k
 *   int bch_encode_bits(int *data_bits, int *codeword_bits) - Encode k bits to n bits
 *   int bch_decode_bits(int *received_bits, int *data_bits) - Decode n bits to k bits, returns errors corrected
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global BCH state
int m, n, length, k, t, d;
int p[21];
int alpha_to[1048576], index_of[1048576], g[1048576];
int data[1048576], bb[1048576], recd[1048576];

// Generate Galois Field
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

// Generate BCH generator polynomial
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

// BCH encoding (systematic)
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

// BCH decoding (Berlekamp-Massey algorithm + Chien search)
int decode_bch() {
    register int i, j, u, q, t2, count = 0, syn_error = 0;
    int **elp;  // Changed to heap allocation
    int *d_array, *l, *u_lu, *s;  // Changed to heap allocation
    int root[200], loc[200], reg[201];
    
    t2 = 2 * t;
    
    // Allocate memory on heap instead of stack
    elp = (int**)malloc(1026 * sizeof(int*));
    for (i = 0; i < 1026; i++) {
        elp[i] = (int*)malloc(1024 * sizeof(int));
    }
    d_array = (int*)malloc(1026 * sizeof(int));
    l = (int*)malloc(1026 * sizeof(int));
    u_lu = (int*)malloc(1026 * sizeof(int));
    s = (int*)malloc(1025 * sizeof(int));
    
    // Form syndromes
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
        // Berlekamp algorithm
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
            // Put elp into index form
            for (i = 0; i <= l[u]; i++)
                elp[u][i] = index_of[elp[u][i]];
            
            // Chien search
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
                count = -1;  // Uncorrectable error
        } else {
            count = -1;  // Too many errors
        }
    }
    
    // Free allocated memory
    for (i = 0; i < 1026; i++) {
        free(elp[i]);
    }
    free(elp);
    free(d_array);
    free(l);
    free(u_lu);
    free(s);
    
    return count;  // Number of corrected errors
}

// ============================================================
// PUBLIC API
// ============================================================

/**
 * Initialize BCH code
 * @param code_m: m parameter (GF(2^m))
 * @param code_length: codeword length n
 * @param code_t: error correction capability
 * @return: k (data bits per codeword), or -1 on error
 */
int bch_init(int code_m, int code_length, int code_t) {
    int i;
    
    m = code_m;
    length = code_length;
    t = code_t;
    n = (1 << m) - 1;
    
    // Set primitive polynomial based on m
    for (i = 1; i < m; i++)
        p[i] = 0;
    p[0] = p[m] = 1;
    
    if (m == 8) {
        p[4] = p[5] = p[6] = 1;
    } else if (m == 7) {
        p[1] = 1;
    } else if (m == 6) {
        p[1] = 1;
    } else if (m == 5) {
        p[2] = 1;
    } else if (m == 4) {
        p[1] = 1;
    } else {
        fprintf(stderr, "Error: Unsupported m value: %d\n", m);
        return -1;
    }
    
    generate_gf();
    gen_poly();
    
    return k;
}

/**
 * Encode k data bits to n codeword bits
 * @param data_bits: input data bits (array of k ints, each 0 or 1)
 * @param codeword_bits: output codeword bits (array of n ints)
 * @return: n (codeword length)
 */
int bch_encode_bits(int *data_bits, int *codeword_bits) {
    int i;
    
    // Copy data bits to global array
    for (i = 0; i < k; i++) {
        data[i] = data_bits[i];
    }
    
    // Encode
    encode_bch();
    
    // Build codeword: [parity | data]
    for (i = 0; i < length - k; i++) {
        codeword_bits[i] = bb[i];
    }
    for (i = 0; i < k; i++) {
        codeword_bits[length - k + i] = data[i];
    }
    
    return length;
}

/**
 * Decode n received bits to k data bits
 * @param received_bits: input received codeword (array of n ints)
 * @param data_bits: output decoded data bits (array of k ints)
 * @return: number of errors corrected, or -1 if uncorrectable
 */
int bch_decode_bits(int *received_bits, int *data_bits) {
    int i, errors;
    
    // Copy received bits to global array
    for (i = 0; i < length; i++) {
        recd[i] = received_bits[i];
    }
    
    // Decode
    errors = decode_bch();
    
    if (errors < 0) {
        return -1;  // Uncorrectable
    }
    
    // Extract data bits from codeword[length-k .. length-1]
    for (i = 0; i < k; i++) {
        data_bits[i] = recd[length - k + i];
    }
    
    return errors;
}
