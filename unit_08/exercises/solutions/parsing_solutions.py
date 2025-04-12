"""
Solutions for Syntactic Parsing Exercises
This module contains solutions for the parsing exercises.
"""

from typing import List, Dict, Tuple, Set
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser
from nltk.tree import Tree
from nltk.parse import ChartParser
from nltk.parse.generate import generate
import spacy
from collections import defaultdict

def exercise_1_chunking() -> Dict[str, List[str]]:
    """Solution for Exercise 1: Chunk Parsing"""
    text = "The skilled programmer wrote efficient code for the new project"
    
    # Define chunk grammar
    grammar = r"""
        NP: {<DT>?<JJ.*>*<NN.*>+}     # Noun phrase
        VP: {<VB.*><NP|PP>}           # Verb phrase
        PP: {<IN><NP>}                # Prepositional phrase
    """
    
    # Create parser and tokenize text
    parser = RegexpParser(grammar)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Parse and extract phrases
    tree = parser.parse(tagged)
    phrases = defaultdict(list)
    
    for subtree in tree.subtrees():
        if subtree.label() in ['NP', 'VP', 'PP']:
            phrase = ' '.join(word for word, tag in subtree.leaves())
            phrases[subtree.label()].append(phrase)
    
    return dict(phrases)

def exercise_2_cfg_parsing() -> List[str]:
    """Solution for Exercise 2: Context-Free Grammar Parsing"""
    # Define CFG
    grammar = nltk.CFG.fromstring("""
        S -> NP VP | Aux NP VP | VP
        NP -> Det N | Det ADJ N | N
        VP -> V | V NP | V NP PP | V PP
        PP -> P NP
        Det -> 'the' | 'a'
        N -> 'cat' | 'dog' | 'program' | 'code'
        V -> 'runs' | 'writes' | 'compiles' | 'debug'
        P -> 'in' | 'on' | 'with'
        ADJ -> 'quick' | 'efficient' | 'buggy'
        Aux -> 'does' | 'can' | 'will'
    """)
    
    # Generate sentences
    sentences = []
    for sentence in generate(grammar, n=5):
        sentences.append(' '.join(sentence))
    
    return sentences

def exercise_3_dependency_analysis(text: str) -> List[Dict[str, str]]:
    """Solution for Exercise 3: Dependency Parsing"""
    # Initialize spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    
    analysis = []
    
    # Analyze each token
    for token in doc:
        # Basic dependency info
        dep_info = {
            'word': token.text,
            'pos': token.pos_,
            'dep': token.dep_,
            'head': token.head.text
        }
        
        # Add subject-verb-object info if applicable
        if token.dep_ == "ROOT":
            subjects = []
            objects = []
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    subjects.append(child.text)
                elif child.dep_ in ["dobj", "pobj"]:
                    objects.append(child.text)
            
            dep_info.update({
                'subjects': subjects,
                'objects': objects
            })
        
        # Add modifier information
        modifiers = [child.text for child in token.children 
                    if child.dep_ in ["amod", "advmod"]]
        if modifiers:
            dep_info['modifiers'] = modifiers
        
        analysis.append(dep_info)
    
    return analysis

def exercise_4_tree_manipulation(tree: Tree) -> Tree:
    """Solution for Exercise 4: Parse Tree Manipulation"""
    def flatten_subtrees(t: Tree, label: str) -> Tree:
        """Flatten subtrees with specified label"""
        if not isinstance(t, Tree):
            return t
        
        if t.label() == label:
            return Tree(t.label(), [leaf for leaf in t.leaves()])
        
        return Tree(t.label(), [flatten_subtrees(child, label) 
                              for child in t])
    
    def simplify_labels(t: Tree) -> Tree:
        """Simplify complex labels to basic categories"""
        if not isinstance(t, Tree):
            return t
        
        # Simplify POS tags
        label = t.label()
        if label.startswith('NN'):
            label = 'N'
        elif label.startswith('VB'):
            label = 'V'
        elif label.startswith('JJ'):
            label = 'ADJ'
        elif label.startswith('RB'):
            label = 'ADV'
        
        return Tree(label, [simplify_labels(child) for child in t])
    
    # Apply transformations
    tree = flatten_subtrees(tree, 'PP')  # Flatten PPs
    tree = simplify_labels(tree)  # Simplify labels
    
    return tree

def exercise_5_parser_evaluation(gold_standard: List[Tree], 
                               predicted: List[Tree]) -> Dict[str, float]:
    """Solution for Exercise 5: Parser Evaluation"""
    def get_constituents(tree: Tree) -> Set[Tuple[str, int, int]]:
        """Extract constituents with their spans"""
        constituents = set()
        for subtree in tree.subtrees():
            if len(subtree.leaves()) > 1:  # Only non-terminal constituents
                leaves = subtree.leaves()
                span = (subtree.label(), 
                       tree.leaves().index(leaves[0]),
                       tree.leaves().index(leaves[-1]) + 1)
                constituents.add(span)
        return constituents
    
    def get_brackets(tree: Tree) -> Set[Tuple[int, int]]:
        """Get bracketing information"""
        brackets = set()
        for subtree in tree.subtrees():
            if len(subtree.leaves()) > 1:
                leaves = subtree.leaves()
                span = (tree.leaves().index(leaves[0]),
                       tree.leaves().index(leaves[-1]) + 1)
                brackets.add(span)
        return brackets
    
    total_precision = 0
    total_recall = 0
    total_crossing = 0
    
    for gold_tree, pred_tree in zip(gold_standard, predicted):
        # Get constituents
        gold_constituents = get_constituents(gold_tree)
        pred_constituents = get_constituents(pred_tree)
        
        # Calculate matches
        matches = len(gold_constituents.intersection(pred_constituents))
        
        # Calculate precision and recall
        precision = matches / len(pred_constituents) if pred_constituents else 0
        recall = matches / len(gold_constituents) if gold_constituents else 0
        
        # Calculate crossing brackets
        gold_brackets = get_brackets(gold_tree)
        pred_brackets = get_brackets(pred_tree)
        
        crossing = 0
        for g_start, g_end in gold_brackets:
            for p_start, p_end in pred_brackets:
                if (g_start < p_start < g_end < p_end or
                    p_start < g_start < p_end < g_end):
                    crossing += 1
        
        total_precision += precision
        total_recall += recall
        total_crossing += crossing
    
    # Calculate final metrics
    num_trees = len(gold_standard)
    avg_precision = total_precision / num_trees
    avg_recall = total_recall / num_trees
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) \
         if (avg_precision + avg_recall) > 0 else 0
    crossing_rate = total_crossing / num_trees
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1,
        'crossing_rate': crossing_rate
    }

def main():
    """Run examples of the solutions"""
    # Example 1: Chunking
    print("\nExample 1: Chunk Parsing")
    chunks = exercise_1_chunking()
    print("\nExtracted phrases:")
    for phrase_type, phrases in chunks.items():
        print(f"\n{phrase_type}:")
        for phrase in phrases:
            print(f"  - {phrase}")
    
    # Example 2: CFG Parsing
    print("\nExample 2: Context-Free Grammar Parsing")
    sentences = exercise_2_cfg_parsing()
    print("\nGenerated sentences:")
    for sentence in sentences:
        print(f"  - {sentence}")
    
    # Example 3: Dependency Analysis
    print("\nExample 3: Dependency Parsing")
    text = "The quick brown fox jumps over the lazy dog"
    analysis = exercise_3_dependency_analysis(text)
    print("\nDependency analysis:")
    for token_info in analysis:
        print(f"\nWord: {token_info['word']}")
        print(f"POS: {token_info['pos']}")
        print(f"Dependency: {token_info['dep']}")
        if 'subjects' in token_info:
            print(f"Subjects: {', '.join(token_info['subjects'])}")
        if 'objects' in token_info:
            print(f"Objects: {', '.join(token_info['objects'])}")
    
    # Example 4: Tree Manipulation
    print("\nExample 4: Tree Manipulation")
    tree = Tree('S', [
        Tree('NP', [Tree('DT', ['the']), Tree('NN', ['cat'])]),
        Tree('VP', [Tree('VBZ', ['chases']), 
                   Tree('NP', [Tree('DT', ['the']), Tree('NN', ['mouse'])])])
    ])
    print("\nOriginal tree:")
    print(tree)
    modified_tree = exercise_4_tree_manipulation(tree)
    print("\nModified tree:")
    print(modified_tree)
    
    # Example 5: Parser Evaluation
    print("\nExample 5: Parser Evaluation")
    gold = [Tree('S', [Tree('NP', ['the', 'cat']), 
                      Tree('VP', ['chases', 'the', 'mouse'])])]
    pred = [Tree('S', [Tree('NP', ['the', 'cat']), 
                      Tree('VP', ['chases', 'the', 'mouse'])])]
    metrics = exercise_5_parser_evaluation(gold, pred)
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    # Run the examples
    main() 