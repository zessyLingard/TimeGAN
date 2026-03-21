/*
 * AES-256-CBC Test - Verify encryption/decryption roundtrip
 * Compile: gcc -o test_aes_cbc test_aes_cbc.c -lssl -lcrypto
 * Run: ./test_aes_cbc
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

#define AES_KEY_SIZE 32
#define AES_IV_SIZE 16
#define AES_BLOCK_SIZE 16

/* AES-256-CBC encrypt (with PKCS7 padding) */
int aes_cbc_encrypt(const unsigned char *plaintext, int plaintext_len,
                    const unsigned char *key, const unsigned char *iv,
                    unsigned char *ciphertext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    
    int len, ciphertext_len;
    
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    
    if (EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len = len;
    
    if (EVP_EncryptFinal_ex(ctx, ciphertext + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    ciphertext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    return ciphertext_len;
}

/* AES-256-CBC decrypt (with PKCS7 unpadding) */
int aes_cbc_decrypt(const unsigned char *ciphertext, int ciphertext_len,
                    const unsigned char *key, const unsigned char *iv,
                    unsigned char *plaintext) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    
    int len, plaintext_len;
    
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    
    if (EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len = len;
    
    if (EVP_DecryptFinal_ex(ctx, plaintext + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return -1;
    }
    plaintext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    return plaintext_len;
}

void print_hex(const char *label, const unsigned char *data, int len) {
    printf("%s (%d bytes): ", label, len);
    for (int i = 0; i < len && i < 32; i++) {
        printf("%02x", data[i]);
    }
    if (len > 32) printf("...");
    printf("\n");
}

int test_aes_cbc(const unsigned char *plaintext, int plaintext_len, const char *test_name) {
    unsigned char key[AES_KEY_SIZE];
    unsigned char iv[AES_IV_SIZE];
    
    /* Generate random key and IV */
    RAND_bytes(key, AES_KEY_SIZE);
    RAND_bytes(iv, AES_IV_SIZE);
    
    /* Calculate max ciphertext size (plaintext + up to 16 bytes padding) */
    int max_ciphertext_len = plaintext_len + AES_BLOCK_SIZE;
    unsigned char *ciphertext = malloc(max_ciphertext_len);
    unsigned char *decrypted = malloc(max_ciphertext_len);
    
    if (!ciphertext || !decrypted) {
        printf("[%s] FAIL: Memory allocation error\n", test_name);
        return -1;
    }
    
    printf("\n=== Test: %s ===\n", test_name);
    print_hex("Plaintext", plaintext, plaintext_len);
    print_hex("Key", key, AES_KEY_SIZE);
    print_hex("IV", iv, AES_IV_SIZE);
    
    /* Encrypt */
    int ciphertext_len = aes_cbc_encrypt(plaintext, plaintext_len, key, iv, ciphertext);
    if (ciphertext_len < 0) {
        printf("[%s] FAIL: Encryption failed\n", test_name);
        free(ciphertext);
        free(decrypted);
        return -1;
    }
    print_hex("Ciphertext", ciphertext, ciphertext_len);
    printf("Padding added: %d bytes\n", ciphertext_len - plaintext_len);
    
    /* Decrypt */
    int decrypted_len = aes_cbc_decrypt(ciphertext, ciphertext_len, key, iv, decrypted);
    if (decrypted_len < 0) {
        printf("[%s] FAIL: Decryption failed\n", test_name);
        free(ciphertext);
        free(decrypted);
        return -1;
    }
    print_hex("Decrypted", decrypted, decrypted_len);
    
    /* Verify */
    if (decrypted_len != plaintext_len) {
        printf("[%s] FAIL: Length mismatch (expected %d, got %d)\n", 
               test_name, plaintext_len, decrypted_len);
        free(ciphertext);
        free(decrypted);
        return -1;
    }
    
    if (memcmp(plaintext, decrypted, plaintext_len) != 0) {
        printf("[%s] FAIL: Content mismatch\n", test_name);
        free(ciphertext);
        free(decrypted);
        return -1;
    }
    
    printf("[%s] PASS: Roundtrip successful!\n", test_name);
    
    free(ciphertext);
    free(decrypted);
    return 0;
}

int main() {
    int failed = 0;
    
    printf("===========================================\n");
    printf("     AES-256-CBC Encryption Test Suite     \n");
    printf("===========================================\n");
    
    /* Test 1: 24 bytes (8 bytes padding) */
    unsigned char test1[] = "Hello, World! 24 bytes!";
    if (test_aes_cbc(test1, 24, "24 bytes (8 padding)") != 0) failed++;
    
    /* Test 2: 31 bytes (1 byte padding - optimal!) */
    unsigned char test2[] = "This is exactly 31 bytes long!";
    if (test_aes_cbc(test2, 31, "31 bytes (1 padding)") != 0) failed++;
    
    /* Test 3: 16 bytes (16 bytes padding - full block) */
    unsigned char test3[] = "Exactly16bytes!";
    if (test_aes_cbc(test3, 16, "16 bytes (16 padding)") != 0) failed++;
    
    /* Test 4: 1 byte (15 bytes padding) */
    unsigned char test4[] = "X";
    if (test_aes_cbc(test4, 1, "1 byte (15 padding)") != 0) failed++;
    
    /* Test 5: 30 bytes (2 bytes padding) */
    unsigned char test5[] = "This message is 30 bytes lon";
    if (test_aes_cbc(test5, 30, "30 bytes (2 padding)") != 0) failed++;
    
    /* Test 6: Binary data with null bytes */
    unsigned char test6[] = "\x00\x01\x02\x03\x00\x05\x06\x07\x08\x09\x00\x0b\x0c\x0d\x0e\x0f"
                            "\x10\x11\x12\x13\x14\x15\x16\x17";
    if (test_aes_cbc(test6, 24, "Binary with nulls") != 0) failed++;
    
    printf("\n===========================================\n");
    if (failed == 0) {
        printf("All tests PASSED! AES-CBC is working correctly.\n");
    } else {
        printf("%d test(s) FAILED!\n", failed);
    }
    printf("===========================================\n");
    
    return failed;
}
