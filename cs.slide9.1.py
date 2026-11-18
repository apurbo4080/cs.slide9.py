# 1. Component classes: a car HAS an engine and brakes.
class Engine:
    def __init__(self, horsepower):
        self.horsepower = horsepower

    def start(self):
        print(f"Engine started ({self.horsepower} HP).")


class Brakes:
    def stop(self):
        print("Brakes applied.")


# 2. Base class: attributes and methods are defined together.
class Car:
    def __init__(self, seats, horsepower):
        self.seats = seats
        self.engine = Engine(horsepower)  # Composition: Car HAS an Engine
        self.brakes = Brakes()            # Composition: Car HAS Brakes

    def drive(self):
        self.engine.start()
        print("The car is moving.")

    def stop(self):
        self.brakes.stop()


# 3. Subclass: a SportsCar IS a Car.
class SportsCar(Car):
    def __init__(self, horsepower):
        super().__init__(seats=2, horsepower=horsepower)

    def drive(self):  # Override the method from Car
        self.engine.start()
        print("The sports car is moving fast!")


# 4. Create objects from the classes (instantiation).
my_car = Car(seats=5, horsepower=150)
sports_car = SportsCar(horsepower=600) 

# 5. Use the objects: each has its own attributes and methods.
print("My car seats:", my_car.seats)
print("Sports car seats:", sports_car.seats)

my_car.drive()       # Car's drive() method
sports_car.drive()   # SportsCar's overridden drive() method
sports_car.stop()    # stop() is inherited from Car
