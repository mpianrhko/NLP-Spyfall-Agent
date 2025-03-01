from data import Location

QUESTIONS = [
    "How often do you come here?",
    "What is there to eat here?",
    "Have you been here before?",
    "Would this location smell good?",
    "Is this place kid-friendly?",
    "Is there a lot of noise in this place?",
    "What time of day is the busiest here?",
    "Do you need a reservation to visit?",
    "What kind of clothing do people typically wear here?",
    "How far is this location from the nearest public transportation?",
    "Are most people here happy?",
]

# Examples answers for the question: "How often do you come here?"
EXAMPLE_ANSWER = {
    Location.AIRPLANE: "A few times a year",
    Location.BANK: "Not often",
    Location.BEACH: "Every once in a while",
    Location.BROADWAY_THEATER: "Not often",
    Location.CASINO: "Not often",
    Location.CATHEDRAL: "Not often",
    Location.CIRCUS_TENT: "Not often",
    Location.CORPORATE_PARTY: "Never",
    Location.CRUSADER_ARMY: "Never",
    Location.DAY_SPA: "Not often",
    Location.EMBASSY: "Never",
    Location.HOSPITAL: "Not often",
    Location.HOTEL: "A few times a year",
    Location.MILITARY_BASE: "Not often",
    Location.MOVIE_STUDIO: "A few times a year",
    Location.OCEAN_LINER: "Never",
    Location.PASSENGER_TRAIN: "Not often",
    Location.PIRATE_SHIP: "Never",
    Location.POLAR_STATION: "Never",
    Location.POLICE_STATION: "Not often",
    Location.RESTAURANT: "A few times a month",
    Location.SCHOOL: "Not often",
    Location.SERVICE_STATION: "Not often",
    Location.SPACE_STATION: "Never",
    Location.SUBMARINE: "Never",
    Location.SUPERMARKET: "A few times a month",
    Location.UNIVERSITY: "Every day",
}


EXAMPLE_BAD_ANSWER = {
    Location.AIRPLANE: "Every day",
    Location.BANK: "Never",
    Location.BEACH: "Every day",
    Location.BROADWAY_THEATER: "Every day",
    Location.CASINO: "Several times a week",
    Location.CATHEDRAL: "Several times a day",
    Location.CIRCUS_TENT: "Once a week",
    Location.CORPORATE_PARTY: "A few times a month",
    Location.CRUSADER_ARMY: "A few times a year",
    Location.DAY_SPA: "Several times a day",
    Location.EMBASSY: "Every week",
    Location.HOSPITAL: "Every day",
    Location.HOTEL: "Never",
    Location.MILITARY_BASE: "Daily",
    Location.MOVIE_STUDIO: "Never",
    Location.OCEAN_LINER: "A few times a week",
    Location.PASSENGER_TRAIN: "A few times a week",
    Location.PIRATE_SHIP: "A few times a week",
    Location.POLAR_STATION: "A few times a year",
    Location.POLICE_STATION: "A few times a week",
    Location.RESTAURANT: "Never",
    Location.SCHOOL: "Never",
    Location.SERVICE_STATION: "Daily",
    Location.SPACE_STATION: "A few times a year",
    Location.SUBMARINE: "Weekly",
    Location.SUPERMARKET: "Never",
    Location.UNIVERSITY: "Never",
}

