
# Building An Object-Oriented Simulation - Lab

## Introduction

In this lab, we'll build a stochastic simulation to model herd immunity in a population, and examine how a virus moves through a population, depending on what percentage of the population is vaccinated against the disease. 

## Objectives

You will be able to:

* Understand the OO lifecycle, and the relationship between attributes and methods
* Create Object-Oriented data models that describe the real world with classes 

<img src='herd_immunity.gif'>

In the previous lesson, we outlined the various steps we'll take to create this simulation.  In this lab, we'll actually build it! 

Since we'll be building a stochastic simulation that makes use of randomness, we'll start by importing numpy and setting a random seed for reproducibility.

Run the cell below to do this now. 


```python
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
np.random.seed(0)
```

## The Assumptions of Our Model

In order to build this stochastic simulation, we'll have to accept some assumptions.  In order to simplify the complexity of the model, we'l assume:

* Vaccines are 100% effective.
* Infected individuals that recover from the disease are now immune to catching the disease a second time (think Chickenpox)
* Dead invidiuals are not contagious. 
* All infections happen from person-to-person interaction
* All individuals interact with the same amount of people each day
* The `r0` value (pronounced _"R-nought"_) is a statistic from the Centers for Disease Control that estimates the average number of people an infected person will infect before they are no longer contagious.  For this value, we assume:
    * That this number is out of 100 people
    * That this statistic is accurate
    
Building simulations is always a trade-off, since the real world is very, very complex.  As we build our simulation, try to think about ways in which we could make our model more realistic by writing it in such a way that it eliminates one of the assumptions above (e.g. generating a float value for vaccine efficacy on a person-by-person level to eliminate our first assumption). 


### Building our `Person` class

We'll start by building out our `Person` class, which will represent the individuals in our population. 

Our `Person` class should have the following attributes:

* `is_alive = True` 
* `is_vaccinated`, a boolean value which will be determined by generating a random value between 0 and 1.  We will then compare this to `(1 - pct_vaccinated)`, a variable that should be passed in at instantiation time.  If the random number is greater, then this attribute should be `True`-- otherwise, `False`
* `is_infected = False`
* `has_been_infected = False`
* `newly_infected = False`

In the cell below, complete the `Person` class.

**_NOTE:_** To generate a random number between 0 and 1, use `np.random.random()`. 


```python
class Person(object):
    
    def __init__(self):
        pass
        
    def get_vaccinated(self, pct_vaccinated):
        pass
```

Great! Since we're using OOP to build this simulation, it makes sense to have each individual `Person` instance take care of certain details, such as determining if they are vaccinated or not.  The `pct_vaccinated` argument is a value we'll pass into the `Simulation` class, which we can then pass along to each `Person` once our population is created to vaccinate the right amount of people.  

## Creating our `Simulation` Class

Creating our `Simulation` class will be a bit more involved, because this class does all the heavy lifting.  We'll handle this piece by piece, and test that everything is working along the way. 

### Writing our `__init__` method

Our init method should take in the following arguments at instantiation time:

* `self` 
* `population_size`
* `disease_name`
* `r0`
* `mortality_rate`
* `total_time_steps`
* `pct_vaccinated`
* `num_initial_infected`

**_Attributes_**

The attributes `self.disease_name`, `self.mortality_rate`, and `self.total_time_steps` should be set to the corresponding arguments passed in at instantiation time. 

The attribute `self.r0` should be set to set to `r0 / 100` to convert the number to a decimal between 0 and 1.

We'll also create attributes for keeping track of what time step the simulation is on, as well as both the current number of infected during this time step and the total number of people that have been infected at any time during the simulation.  For these, set `self.current_time_step`, `self._total_infected_counter`, and `self.current_infected_counter` to  `0`.

We'll also need to create an array to hold all of the `Person` objects in our simulation.  Set `self.population` equal to an empty list.  


Now comes the fun part--creating the population, and determining if they are healthy, vaccinated, or infected.  

Follow the instructions inside the `__init__` method to write the logic that will set up our `Simulation` correctly. 



```python
class Simulation(object):
    
    def __init__(self, population_size, disease_name, r0, mortality_rate,  total_time_steps, pct_vaccinated, num_initial_infected):
        self.r0 = None
        self.disease_name = None
        self.mortality_rate = None
        self.total_time_steps = None
        self.current_time_step = None
        self.total_infected_counter = None
        self.current_infected_counter = None
        self.dead_counter = None
        self.population = None
        # This attribute is used in a function that is provided for you in order to log statistics from each time_step.  
        # Don't touch it!
        self.time_step_statistics_df = pd.DataFrame()
        
        # Create a for loop the size of the population we want in this simulation
        for i in range(None):
            # Create new person
            new_person = None
            # We'll add infected persons to our simulation first.  Check if the current number of infected are equal to the 
            # num_initial_infected parameter.  If not, set new_person to be infected
            if self.current_infected_counter != None:
                new_person.is_infected = None
                # dont forget to increment both infected counters!
                self.total_infected_counter += None
                self.current_infected_counter += None
            # if new_person is not infected, determine if they are vaccinated or not by using their `get_vaccinated` method
            # Then, append new_person to self.population
            else:
                
            self.population.append(None)
       
        print("-" * 50)
        print("Simulation Initiated!")
        print("-" * 50)
        self._get_sim_statistics()
        
        
    
    def _get_sim_statistics(self):
    # In the interest of time, this method has been provided for you.  No extra code needed.
        num_infected = 0
        num_dead = 0
        num_vaccinated = 0
        num_immune = 0
        for i in self.population:
            if i.is_infected:
                num_infected += 1
            if not i.is_alive:
                num_dead += 1
            if i.is_vaccinated:
                num_vaccinated += 1
                num_immune += 1
            if i.has_been_infected:
                num_immune += 1
        assert num_infected == self.current_infected_counter
        assert num_dead == self.dead_counter
        
        
        print("")
        print("Summary Statistics for Time Step {}".format(self.current_time_step))
        print("")
        print("-" * 50)
        print("Disease Name: {}".format(self.disease_name))
        print("R0: {}".format(self.r0 * 100))
        print("Mortality Rate: {}%".format(self.mortality_rate * 100))
        print("Total Population Size: {}".format(len(self.population)))
        print("Total Number of Vaccinated People: {}".format(num_vaccinated))
        print("Total Number of Immune: {}".format(num_immune))
        print("Current Infected: {}".format(num_infected))
        print("Deaths So Far: {}".format(num_dead))     
```

Great! We've now created a basic `Simulation` object that is capable of instantiating itself according to our specifications.  However, our simulation doesn't currently do anything.  Now, we'll add the appropriate behaviors to our simulation. 

### Building Our Simulation's Behavior

For any given time step, our simulation should complete the following steps in order:

1. Loop through each living person in the population
    1A. If the person is currently infected:
        1B. Select another random person from the population. 
        2B. If this person is alive, not infected, unvaccinated, and hasn't been infected before: 
            1C. Generate a random number between 0 and 1.  If this random number is greater than `(1 - self.r0)`, then mark this new person as newly infected
        3B.If the person is vaccinated, currently infected, or has been infected in a previous round of the simulation, do nothing. 
    2A. Repeat the step above until the infected person has interacted with 100 random living people from the population. 
2. Once every infected person has interacted with 100 random living people, resolve all current illnesses and new infections
    2A. For each person that started this round as infected, generate a random number between 0 and 1.  If that number is greater than `(1 - mortality rate)`, then that person has been killed by the disease.  They should be marked as dead.  Otherwise, they stay alive, and can longer catch the disease.
    2B.  All people that were infected this round move from `newly_infected` to `is_infected`. 
    
We'll begin by breaking up most of this logic into helper functions, so that our main functions will be simple.  
    
#### `infected_interaction()` Function

We'll begin by writing a function called `infected_interaction` that will be called for every infected person in the population in a given time step.  This function handles all the possible cases that can happen with the following logic:

* Initialize a counter called `num_interactions` to `0`.
* Select a random person from `self.population`.
* Check is the person is alive. If the person is dead, we will not count this as an interaction.
* If the random person is alive and not vaccinated, generate a a random number between 0 and 1.  If the random number is greater than `(1 - self.r0)`, change the random person's `newly_infected` attribute to `True`.
* Increment `num_interactions` by 1.  Do not increment any of the infected counters in the simulation class--we'll have another method deal with those.

Complete the `infected_interaction()` method in the cell below.  Comments have been provided to help you write it.

**_HINT_**: To randomly select an item from a list, use `np.random.choice()`!


```python
def infected_interaction(self, infected_person):
    num_interactions = None
    while num_interactions < 100:
        # Randomly select a person from self.population
        random_person = None
        # This only counts as an interaction if the random person selected is alive.  If the person is dead, we do nothing, 
        # and the counter doesn't increment, repeating the loop and selecting a new person at random.
        # check if the person is alive.
        if None:
            # CASE: Random person is not vaccinated, and has not been infected before, making them vulnerable to infection
            if None and None:
                # Generate a random number between 0 and 1
                random_number = None
                # If random_number is greater than or equal to (1 - self.r0), set random person as newly_infected
                if None >= None:
                    random_person.newly_infected = None
            # Dont forget to increment num_interactions, and make sure it's at this level of indentation
            num_interactions += None

# Adds this function to our Simulation class
Simulation.infected_interaction = infected_interaction
```

#### `_resolve_states()` Function

The 2nd helper function we'll use during each time step is one that resolves any temporary states.  Recall that people do not stay newly infected or infected for more than a turn.  That means we need a function to figure out what happens to these people at the end of each turn, so that everything is ready to go for the next time step. 

This function will:

* Iterate through every person in the population.
* Check if the person is alive (since we dont need to bother checking anything for the dead ones)
* If the person is infected, we need to resolve whether they survive the infection or die from it.  
    * Generate a random number between 0 and 1.  
    * If this number is greater than `(1 - self.mortality_rate)`, the person has died.  
        * Set the person's `.is_alive` and `.is_infected` attributes both to `False`.  
        * Increment the simulation's `self.dead_counter` attribute by 1.
        * Decrement the simulation's `self.current_infected_counter` attribute by 1.
    * Else, the person has survived the infection and is now immune to future infections.
        * Set the person's `is_infected` attribute to `False`
        * Set the person's `has_been_infected` attribute to `True`
        * Decrement the simulation's `self.current_infected_counter` by 1.
* If the person is newly infected:
    * Set the person's `newly_infected` attribute to `False`
    * Set the person's `is_infected` attribute to `True`
    * Increment `total_infected_counter` and `current_infected_counter` by 1.
    

Complete the function `_resolve_states()` in the cell below.  Comments have been provided to help you write it. 


```python
def _resolve_states(self):
    """
    Every person in the simulation falls into 1 of 4 states at any given time:
    1. Dead 
    2. Alive and not infected
    3. Currently infected
    4. Newly Infected
    
    States 1 and 2 need no resolving, but State 3 will resolve by either dying or surviving the disease, and State 4 will resolve
    by turning from newly infected to currently infected.
    
    This method will be called at the end of each time step.  All states must be resolved before the next time step can begin.
    """
    # Iterate through each person in the population
    for person in None:
        # We only need to worry about the people that are still alive
        if None: 
            # CASE: Person was infected this round.  We need to stochastically determine if they die or recover from the disease
            # Check if person is_infected
            if None:
                # Generate a random number
                random_number = None
                # If random_number is >= (1 - self.mortality_rate), set the person to dead and increment the simulation's death
                # counter
                if None >= None:
                    # Set is_alive and in_infected both to False
                    person.is_alive = None
                    person.is_infected = None
                    # Don't forget to increment self.dead_counter, and decrement self.current_infected_counter
                    self.dead_counter += None
                    self.current_infected_counter -= None
                else:
                    # CASE: They survive the disease and recover.  Set is_infected to False and has_been_infected to True
                    person.is_infected = None
                    person.has_been_infected = None
                    # Don't forget to decrement self.current_infected_counter!
                    self.current_infected_counter -= None
            # CASE: Person was newly infected during this round, and needs to be set to infected before the start of next round
            elif None:
                # Set is_infected to True, newly_infected to False, and increment both self.current_infected_counter and 
                # self.total_infected_counter
                person.is_infected = None
                person.newly_infected = None
                self.current_infected_counter += None
                self.total_infected_counter += None
                
Simulation._resolve_states = _resolve_states
```

#### `_time_step()` Function

Now that we have two helper methods to most of the heavy lifting for us, we'll find that our `_time_step()` function will be pretty simple.  

This function should:

* Iterate through each person in the population
* If the person is alive and infected, call `self.infected_interaction()` and pass in this infected person
* Once we have looped through every person, call `self._resolve_states()` to resolve all outstanding states and prepare for the next round.  
* Log the statistics from this round by calling `self._log_time_step_statistics()`.  This function has been provided for you further down the notebook. 
* Increment `self.current_time_step`.


```python
def _time_step(self):
    """
    Compute 1 time step of the simulation. This function will make use of the helper methods we've created above. 
    
    The steps for a given time step are:
    1.  Iterate through each person in self.population.
        - For each infected person, call infected_interaction() and pass in that person.
    2.  Use _resolve_states() to resolve all states for the newly infected and the currently infected.
    3. Increment self.current_time_step by 1. 
    """
    # Iterate through each person in the population
    for person in None:
        # Check only for people that are alive and infected
        if None and None:
            # Call self.infected_interaction() and pass in this infected person
            None
    
    # Once we've made it through the entire population, call self._resolve_states()
    None
    
    # Now, we're almost done with this time step.  Log summary statistics, and then increment self.current_time_step by 1.
    None
    self.current_time_step += None

# Adds this function to our Simulation class
Simulation._time_step = _time_step
```

Finally, we just need to write a function that logs the results of each time step by storing it in a DataFrame, and then writes the end result of the Simulation to a csv file.

In the interest of time, this function has been provided for you.  You do not need to write any code in the cell below--just run the cell.


```python
def _log_time_step_statistics(self, write_to_file=False):
    # This function has been provided for you, you do not need to write any code for it.
    # Gets the current number of dead,
    # CASE: Round 0 of simulation, need to create and Structure DataFrame
#     if self.time_step_statistics_df == None:
#         import pandas as pd
#         self.time_step_statistics_df = pd.DataFrame()
# #         col_names = ['Time Step', 'Currently Infected', "Total Infected So Far" "Alive", "Dead"]
# #         self.time_step_statistics_df.columns = col_names
#     # CASE: Any other round
#     else:
        # Compute summary statistics for currently infected, alive, and dead, and append them to time_step_snapshots_df
    row = {
        "Time Step": self.current_time_step,
        "Currently Infected": self.current_infected_counter,
        "Total Infected So Far": self.total_infected_counter,
        "Alive": len(self.population) - self.dead_counter,
        "Dead": self.dead_counter
    }
    self.time_step_statistics_df = self.time_step_statistics_df.append(row, ignore_index=True)
    
    if write_to_file:
        self.time_step_statistics_df.to_csv("simulation.csv", mode='w+')
        
Simulation._log_time_step_statistics = _log_time_step_statistics
```

#### Wrapping It All Up With `run()`

This is the function that our user will actually interact with. It will act as our "main" function.  Since we've done a great job of writing very clean, modular helper functions, we'll find that this function is the simplest one in the entire program. 

This function should:

* Start a for loop that runs `self.total_time_steps` number of times.  In order to demonstrate how to easily add a progress bar to iterables with the `tqdm` library, this line has been added for you. 
* Display a message telling the user the time step that it is currently working on. 
* Call `self._time_step()`
* Once the simuluation has finished, write the DataFrame containing the summary statistics from each step to a csv file.  This line of code has also been provided for you. 

In the cell below, complete the `run()` function.  Comments have been added to help you write it. 


```python
def run(self):
    """
    The main function of the simulation.  This will run the simulation starting at time step 0, calculating
    and logging the results of each time step until the final time_step is reached. 
    """
    
    for _ in tqdm(range(self.total_time_steps)):
        # Print out the current time step 
        print("Beginning Time Step {}".format(None))
        # Call our `_time_step()` function
        None
    
    # Simulation is over--log results to a file by calling _log_time_step_statistics(write_to_file=True)
    self._log_time_step_statistics(write_to_file=True)

# Adds the run() function to our Simulation class.
Simulation.run = run
```

### Running Our Simulation

Now comes the fun part--actually running our simulation!

In the cell below, create a simulation with the following parameters:

* Population size of `2000`
* Disease name is `Ebola`
* r0 value of `2`
* Mortality rate of `0.5`
* `20` time steps
* Vaccination Rate of `0.85`
* `50` initial infected


```python
sim = None
```


```python
# Call sim.run() below!

```


```python
results = pd.read_csv('simulation.csv')
results
```

If you didn't change the random seed, your results should look like this:

<img src='example_sim_results.png'>

As we can see from the table above, even though the average person with Ebola will infect 2 other people, herd immunity protects 
the majority of the unvaccinated people in the population.  Although there were approximately 400 people in the population that were susceptible to Ebola, only 67 were actually infected before the virus burned itself out!

## Extra Practice

Try different values for the `pct_vaccinated` argument, and see how it changes the results of the model.  These would look great as a visualization--consider comparing them all on the same line graph!

## Summary

Great job! You've just written a simulation to demonstrate the effects of herd immunity in action. 

In this lab, we demonstated mastery of:

* Understanding the OO lifecycle , and the relationship between attributes and methods
* Creating Object-Oriented data models that describe the real world with classes
