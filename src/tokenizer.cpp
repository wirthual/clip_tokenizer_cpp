#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <climits>
#include <iomanip>
#include <zlib.h>
#include <regex>

SimpleTokenizer::SimpleTokenizer() 
    : bpe_ranks() {
    initialize_byte_encoder();

    initialize_vocabulary();
    
    // Pattern for tokenization
    pat = std::regex(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]+)", 
                     std::regex::icase);
}


void SimpleTokenizer::initialize_byte_encoder() {
    std::vector<int> bs;

    // "!" to "~"  (33 to 126)
    for (int c = '!'; c <= '~'; ++c)
        bs.push_back(c);

    // "¡" to "¬"  (161 to 172)
    for (int c = 0xA1; c <= 0xAC; ++c)
        bs.push_back(c);

    // "®" to "ÿ"  (174 to 255)
    for (int c = 0xAE; c <= 0xFF; ++c)
        bs.push_back(c);

    byte_encoder.resize(256);
    
    // Map each byte to a string representation, matching Python's bytes_to_unicode()
    // For printable bytes, use the character itself
    // For non-printable, map to chr(256 + offset)
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) != bs.end()) {
            // Printable: use character itself
            byte_encoder[b] = std::string(1, static_cast<unsigned char>(b));
        } else {
            // Non-printable: map to Unicode char 256+n
            int unicode_val = 256 + n;
            // Encode as UTF-8
            std::string utf8;
            if (unicode_val < 0x80) {
                utf8 = std::string(1, static_cast<char>(unicode_val));
            } else if (unicode_val < 0x800) {
                utf8 += static_cast<char>(0xC0 | (unicode_val >> 6));
                utf8 += static_cast<char>(0x80 | (unicode_val & 0x3F));
            } else {
                utf8 += static_cast<char>(0xE0 | (unicode_val >> 12));
                utf8 += static_cast<char>(0x80 | ((unicode_val >> 6) & 0x3F));
                utf8 += static_cast<char>(0x80 | (unicode_val & 0x3F));
            }
            byte_encoder[b] = utf8;
            n += 1;
        }
    }
    
    // Create reverse mapping
    for (int i = 0; i < 256; ++i) {
        byte_decoder[byte_encoder[i]] = i;
    }
}

void SimpleTokenizer::initialize_vocabulary() {
    // Initialize vocabulary matching Python's bytes_to_unicode() dict insertion order
    // Python's vocab is built from dict.values() which maintains insertion order
    // The dict is built by zipping bytes in order [33-126, 161-172, 174-255, then unmapped] with chars
    
    std::vector<std::string> vocab;
    
    // Build list of bytes in the same order Python uses
    std::vector<int> ordered_bytes;
    
    // "!" to "~"  (33 to 126)
    for (int c = 33; c <= 126; ++c)
        ordered_bytes.push_back(c);

    // "¡" to "¬"  (161 to 172)
    for (int c = 0xA1; c <= 0xAC; ++c)
        ordered_bytes.push_back(c);

    // "®" to "ÿ"  (174 to 255)
    for (int c = 0xAE; c <= 0xFF; ++c)
        ordered_bytes.push_back(c);

    // Then unmapped bytes (0-32, 127-160, 173)
    for (int b = 0; b < 256; ++b) {
        if (std::find(ordered_bytes.begin(), ordered_bytes.end(), b) == ordered_bytes.end()) {
            ordered_bytes.push_back(b);
        }
    }
    
    // Build vocab from byte_encoder in this order
    for (int b : ordered_bytes) {
        vocab.push_back(byte_encoder[b]);
    }
    
    // Add all with </w> suffix
    for (int b : ordered_bytes) {
        vocab.push_back(byte_encoder[b] + "</w>");
    }

    std::string vocab_path = "./bpe_simple_vocab_16e6.txt.gz";
#ifdef CTC_DATA_DIR
    vocab_path = std::string(CTC_DATA_DIR) + "/bpe_simple_vocab_16e6.txt.gz";
#endif
    merges = read_gzip_lines(vocab_path);
    merges = {merges.begin() + 1, merges.begin() + 48895};

    // Process merges
    for (const std::string& merge : merges) {
        auto parts = splitString(merge, ' ');
        if (parts.size() == 2) {
            merges_splits.push_back(parts);
            // Add merged token to vocab
            vocab.push_back(parts[0] + parts[1]);
            // Map the pair to its rank
            bpe_ranks[{parts[0], parts[1]}] = merges_splits.size() - 1;
        }
    }
    
    // Add special tokens
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");
    
    // Build encoder/decoder
    for (size_t i = 0; i < vocab.size(); ++i) {
        encoder[vocab[i]] = i;
        decoder[i] = vocab[i];
    }
    
    // Initialize cache with special tokens
    cache["<|startoftext|>"] = "<|startoftext|>";
    cache["<|endoftext|>"] = "<|endoftext|>";
}

std::string SimpleTokenizer::basic_clean(const std::string& text) {
    // Simplified: just return trimmed text
    std::string result = text;
    // Remove leading/trailing whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    return result;
}

std::string SimpleTokenizer::whitespace_clean(const std::string& text) {
    std::string result = text;
    // Replace multiple spaces with single space
    result = std::regex_replace(result, std::regex(R"(\s+)"), " ");
    // Trim
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    return result;
}

std::set<std::pair<std::string, std::string>> SimpleTokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs.insert({word[i], word[i + 1]});
    }
    return pairs;
}

std::string SimpleTokenizer::bpe(const std::string& token) {
    // Check cache first
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }
    
    // Convert token to vector of characters with </w> suffix
    std::vector<std::string> word;
    for (size_t i = 0; i < token.length(); ++i) {
        word.push_back(std::string(1, token[i]));
    }
    // Add end-of-word marker to last character
    if (!word.empty()) {
        word.back() += "</w>";
    }
    
    // Get initial pairs
    auto pairs = get_pairs(word);
    
    if (pairs.empty()) {
        // No pairs to merge, just join with spaces and cache
        std::string result;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i > 0) result += " ";
            result += word[i];
        }
        cache[token] = result;
        return result;
    }
    
    // Iteratively merge most frequent pairs
    while (true) {
        // Find the bigram with the lowest rank (most frequent merge)
        auto bigram_iter = pairs.begin();
        int min_rank = INT_MAX;
        std::pair<std::string, std::string> best_bigram = *bigram_iter;
        
        for (const auto& pair : pairs) {
            auto rank_iter = bpe_ranks.find(pair);
            int rank = (rank_iter != bpe_ranks.end()) ? rank_iter->second : INT_MAX;
            
            if (rank < min_rank) {
                min_rank = rank;
                best_bigram = pair;
            }
        }
        
        // If this bigram doesn't exist in bpe_ranks, we're done
        if (min_rank == INT_MAX) {
            break;
        }
        
        std::string first = best_bigram.first;
        std::string second = best_bigram.second;
        
        // Merge all occurrences of this bigram in the word
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            // Find next occurrence of first element starting from i
            size_t j = word.size();
            for (size_t k = i; k < word.size(); ++k) {
                if (word[k] == first) {
                    j = k;
                    break;
                }
            }
            
            // Add everything between i and j
            for (size_t k = i; k < j; ++k) {
                new_word.push_back(word[k]);
            }
            
            if (j == word.size()) {
                break;
            }
            
            i = j;
            
            // Check if we can merge at position i
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        
        word = new_word;
        
        // If only one token left, we're done
        if (word.size() == 1) {
            break;
        }
        
        // Recompute pairs for next iteration
        pairs = get_pairs(word);
    }
    
    // Join word elements with spaces and cache
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += " ";
        result += word[i];
    }
    
    cache[token] = result;
    return result;
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) {
    std::vector<int> bpe_tokens;
        
        // Clean and lowercase
        std::string cleaned = whitespace_clean(basic_clean(text));
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        
        // Find all pattern matches
        std::sregex_iterator iter(cleaned.begin(), cleaned.end(), pat);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::string token = iter->str();
            std::cout << token << std::endl;
            
            // Encode token to UTF-8 bytes, then map through byte_encoder
            auto utf8_bytes = utf8_encode(token);
            std::string encoded_token;
            for (uint8_t byte : utf8_bytes) {
                encoded_token += byte_encoder[byte];
            }
            
            // Apply BPE and collect token IDs
            std::string bpe_result = bpe(encoded_token);
            std::istringstream iss(bpe_result);
            std::string bpe_token;
            while (iss >> bpe_token) {
                if (encoder.count(bpe_token)) {
                    bpe_tokens.push_back(encoder[bpe_token]);
                }
            }
        }
        
        return bpe_tokens;
    }

std::string SimpleTokenizer::decode(const std::vector<int>& tokens) {
    // Simply reconstruct by looking up each token in the decoder
    std::string text;
    for (int token : tokens) {
        if (decoder.find(token) != decoder.end()) {
            std::string word = decoder[token];
            word = std::regex_replace(word, std::regex("</w>"), " ");
            text += word;
        }
    }
    return text;
}


std::vector<std::string> SimpleTokenizer::read_gzip_lines(const std::string& path) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Failed to open gzip file");

    std::string content;
    char buffer[4096];

    int bytes;
    while ((bytes = gzread(file, buffer, sizeof(buffer))) > 0) {
        content.append(buffer, bytes);
    }

    gzclose(file);

    // Split by newline
    std::vector<std::string> lines;
    std::stringstream ss(content);
    std::string line;

    while (std::getline(ss, line)) {
        lines.push_back(line);
    }

    return lines;
}


std::vector<std::string> SimpleTokenizer::splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(str);

    // Read characters from ss into token until the delimiter is found
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}
