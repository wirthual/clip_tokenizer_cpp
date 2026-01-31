#ifndef SIMPLE_TOKENIZER_H
#define SIMPLE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <regex>
#include <memory>
#include <functional>
#include <iostream>
#include <zlib.h>
#include <string>
#include <vector>
#include <sstream>


class SimpleTokenizer {
public:
    SimpleTokenizer();
    
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    
private:
    std::vector<std::string> byte_encoder;  // byte_encoder[i] = encoded form of byte i
    std::map<std::string, int> byte_decoder; // byte_decoder[encoded] = original byte
    std::vector<std::string> merges;
    std::vector<std::vector<std::string>> merges_splits;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    std::unordered_map<std::string, std::string> cache;
    std::regex pat;
    
    void initialize_byte_encoder();
    void initialize_vocabulary();

    std::vector<std::string> read_gzip_lines(const std::string& path);
    
    std::string basic_clean(const std::string& text);
    std::string whitespace_clean(const std::string& text);
    
    std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
    std::vector<std::string> bytes_to_unicode_chars(const std::string& token);
    
    std::string bpe(const std::string& token);

    std::vector<std::string> splitString(const std::string& str, char delimiter);

    std::vector<uint8_t> utf8_encode(const std::string& str) {
        std::vector<uint8_t> bytes;
        for (unsigned char c : str) {
            bytes.push_back(c);
        }
        return bytes;
    }

};

#endif // SIMPLE_TOKENIZER_H
