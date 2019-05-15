class ShoppingCart:
    
    def __init__(self, emp_discount=None):
        self.total = 0
        self.emp_discount = emp_discount
        self.items = []
       
    def add_item(self, name, price, quantity=1):
        for item in list(range(quantity)):
            self.items.append({'name_of_item':name, 'price_of_item':price})
            self.total = self.total+price   
        print(self.total)
        print(self.items)
 
    def mean_item_price(self):
        return self.total/len(self.items)

    def median_item_price(self):
        list_for_median = [item["price_of_item"] for item in self.items]
        import statistics
        return statistics.median(list_for_median)
     
    def apply_discount(self):
        if self.emp_discount == None:
            return("Sorry, there is no discount!")
        else:
            return (self.total-(self.total*self.emp_discount/100))

    def void_last_item(self):
        if len(self.items) == 0:
            return 'Your cart is empty!'
        else:
            self.items.pop()
            return self.items