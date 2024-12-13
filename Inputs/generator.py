import random
import sys


def generate_flight_data(num_flights, num_airports, output_file):
    """
    Generates a random flight schedule dataset ensuring the total number of airplanes
    across all airports does not exceed the number of flights and departure timestep < arrival timestep.

    Args:
    - num_flights: Number of flights to generate.
    - num_airports: Number of airports to include in the dataset.
    - output_file: File to save the generated dataset.
    """
    airports = [f"AP{i}" for i in range(num_airports)]  # Generate unique airport codes
    days = [0, 1, 2]  # Example range of days

    total_planes = 0
    airplane_distribution = []

    # Distribute airplanes randomly across airports, ensuring the sum <= num_flights
    for i in range(num_airports):
        if i == num_airports - 1:
            airplanes_at_airport = num_flights - total_planes  # Assign remaining planes to the last airport
        else:
            airplanes_at_airport = random.randint(1, max(1, (num_flights - total_planes) // (num_airports - i)))
        total_planes += airplanes_at_airport
        airplane_distribution.append((airports[i], airplanes_at_airport))

    if total_planes > num_flights:
        print("Error: Total airplanes exceed the number of flights!")
        sys.exit(1)

    with open(output_file, 'w') as file:
        # Write header with number of flights and airports with airplanes
        file.write(f"{num_flights} {num_airports}\n")

        # Generate flight details
        for _ in range(num_flights):
            depart_airport = random.choice(airports)
            arrive_airport = random.choice([a for a in airports if a != depart_airport])  # Ensure different airports

            depart_day = random.choice(days)
            depart_time = random.randint(0, 23)

            # Ensure departure timestep < arrival timestep
            min_arrival_timestep = depart_day * 24 + depart_time + 1  # At least 1 hour after departure
            max_arrival_timestep = min_arrival_timestep + 6  # Maximum flight duration of 6 hours

            arrive_timestep = random.randint(min_arrival_timestep, max_arrival_timestep)
            arrive_day = arrive_timestep // 24
            arrive_time = arrive_timestep % 24

            file.write(f"{depart_airport} {depart_day} {depart_time} {arrive_airport} {arrive_day} {arrive_time}\n")

        # Write airport airplane distributions
        for airport, airplanes in airplane_distribution:
            file.write(f"{airport} {airplanes}\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_flight_data.py <num_flights> <num_airports> <output_file>")
        sys.exit(1)

    try:
        num_flights = int(sys.argv[1])
        num_airports = int(sys.argv[2])
        output_file = sys.argv[3]

        if num_airports > num_flights:
            print("Error: Number of airports cannot exceed the number of flights!")
            sys.exit(1)

    except ValueError:
        print("Please provide valid integers for the number of flights and airports.")
        sys.exit(1)

    generate_flight_data(num_flights, num_airports, output_file)
    print(f"Flight data generated and saved to {output_file}")
