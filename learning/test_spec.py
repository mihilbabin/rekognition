
# Python code to demonstrate working of unittest 
import unittest 
  
class TestStringMethods(unittest.TestCase): 
      
    def setUp(self): 
        pass
  
    # Returns True if the string contains 4 a. 
    def test_strings_a(self): 
        self.assertEqual( 'a'*4, 'aaaa') 
  
    # Returns True if the string is in upper case. 
    def test_upper(self):         
        self.assertEqual('foo'.upper(), 'FOO') 
  
    # Returns TRUE if the string is in uppercase 
    # else returns False. 
    def test_isupper(self):         
        self.assertTrue('FOO'.isupper()) 
        self.assertFalse('Foo'.isupper()) 
  
    # Returns true if the string is stripped and  
    # matches the given output. 
    def test_strip(self):         
        s = 'geeksforgeeks'
        self.assertEqual(s.strip('geek'), 'sforgeeks') 
  
    # Returns true if the string splits and matches 
    # the given output. 
    def test_split(self):         
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world']) 
        with self.assertRaises(TypeError): 
            s.split(2) 
    
    def test_lstrip(self): #testing for left stripping
        self.assertEqual('   hello '.lstrip(),'hello ')

    def test_isupper(self): #testing for isupper
        self.assertTrue('HELLO'.isupper())
        self.assertFalse('HELlO'.isupper())
    
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 3)), 6, "Should be 6")
  
if __name__ == '__main__': 
    unittest.main() 