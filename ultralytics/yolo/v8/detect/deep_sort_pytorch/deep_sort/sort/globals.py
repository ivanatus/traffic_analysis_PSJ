# -*- coding: utf-8 -*-
"""
Create globals.py for retrieving information about frames to import in predict and detection

this file is in the same folder as detection.py
"""

class Globals:
    
    global_frame = 0
    no_of_people = 0
    no_of_cars = 0
    no_of_bikes = 0
    no_of_buses = 0
    no_of_trucks = 0
    no_of_trains = 0
    
    @staticmethod
    def set_global_frame(frame):
        Globals.global_frame = frame
        
    @staticmethod
    def get_global_frame():
        return Globals.global_frame
    
    @staticmethod
    def set_no_of_people(n):
        Globals.no_of_people = n
        print("People " + str(Globals.no_of_people))
        
    @staticmethod
    def get_no_of_people():
        return Globals.no_of_people
    
    @staticmethod
    def set_no_of_cars(n):
        Globals.no_of_cars = n
        print("Cars " + str(Globals.no_of_cars))
        
    @staticmethod
    def get_no_of_cars():
        return Globals.no_of_cars
    
    @staticmethod
    def set_no_of_bikes(n):
        Globals.no_of_bikes = n
        print("Bikes " + str(Globals.no_of_bikes))
        
    @staticmethod
    def get_no_of_bikes():
        return Globals.no_of_bikes
    
    @staticmethod
    def set_no_of_buses(n):
        Globals.no_of_buses = n
        print("Buses " + str(Globals.no_of_buses))
        
    @staticmethod
    def get_no_of_buses():
        return Globals.no_of_buses
    
    @staticmethod
    def set_no_of_trucks(n):
        Globals.no_of_trucks = n
        print("Trucks " + str(Globals.no_of_trucks))
        
    @staticmethod
    def get_no_of_trucks():
        return Globals.no_of_trucks
    
    @staticmethod
    def set_no_of_trains(n):
        Globals.no_of_trains = n
        print("Trains " + str(Globals.no_of_trains))
        
    @staticmethod
    def get_no_of_trains():
        return Globals.no_of_trains