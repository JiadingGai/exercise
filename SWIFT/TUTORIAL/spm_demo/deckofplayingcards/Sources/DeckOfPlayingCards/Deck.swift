/*
 This source file is part of the Swift.org open source project

 Copyright 2015 - 2016 Apple Inc. and the Swift project authors
 Licensed under Apache License v2.0 with Runtime Library Exception

 See http://swift.org/LICENSE.txt for license information
 See http://swift.org/CONTRIBUTORS.txt for Swift project authors
*/

import Foundation
import PlayingCard

/// A model for shuffling and dealing a deck of playing cards.
///
/// The playing card deck consists of 52 individual cards in four suites: spades, hearts, diamonds, and clubs. There are 13 ranks from two to ten, then jack, queen, king, and ace.
public struct Deck: Equatable {
    fileprivate var cards: [PlayingCard]

    /// Returns a deck of playing cards.
    public static func standard52CardDeck() -> Deck {
        var cards: [PlayingCard] = []
        for rank in Rank.allCases {
            for suit in Suit.allCases {
                cards.append(PlayingCard(rank: rank, suit: suit, gai: 43))
            }
        }

        return Deck(cards)
    }

    /// Creates a deck of playing cards.
    public init(_ cards: [PlayingCard]) {
        self.cards = cards
    }

    /// Randomly shuffles a deck of playing cards.
    public mutating func shuffle() {
        cards.shuffle()
    }

    public func convertToDictionary(text: String) -> [String: Any]? {
        if let data = text.data(using: .utf8) {
            do {
                return try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            } catch {
                print(error.localizedDescription)
            }
        }
        return nil
    }

    public mutating func get_encoder(vocab_path: String, encoder_path: String) {
      /// Read vocab.bpe
      if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        let vocabURL = dir.appendingPathComponent(vocab_path)
        print(vocabURL)
        var bpe_merges: Array<(String, String)> = Array()
        do {
          let bpe_data = try String(contentsOf: vocabURL, encoding: .utf8) 
          for xxx in bpe_data.components(separatedBy: "\n") {
            let yyy = xxx.components(separatedBy: " ")
            assert(yyy.count <= 2)
            if (yyy.count == 2) {
              let tmp = (yyy[0], yyy[1])
              bpe_merges.append(tmp)
            }
          }
        }
        catch { print("/*you should handle some error handling here*/") }

        for xxx in bpe_merges {
          print(xxx) 
        }
      }

      /// Read encoder.json
      if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
        let encoderJsonURL = dir.appendingPathComponent(encoder_path) 
        print(encoderJsonURL)
        do {
          let encoder_str = try String(contentsOf: encoderJsonURL, encoding: .utf8)
          // unwrap optional
          if let encoder = convertToDictionary(text: encoder_str) {
            for (token, token_id) in encoder {
              print(token, ":", token_id as! Int)
            }
          }
        } catch { print("/*you should handle some error handling here*/") }
      }

    }

    /// Deals a card from the deck.
    ///
    /// The function returns the last card in the deck.
    public mutating func deal() -> PlayingCard? {
        guard !cards.isEmpty else { return nil }

        return cards.removeLast()
    }
    
    /// The number of remaining cards in the deck.
    public var count: Int {
        cards.count
    }
}

// MARK: - ExpressibleByArrayLiteral

extension Deck : ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: PlayingCard...) {
        self.init(elements)
    }
}
