from datetime import datetime, timedelta, time
import pytz
from gurobipy import Model, GRB, quicksum
from icalendar import Calendar, Event
import pandas as pd


def get_timezone(timezone_str):
    """
    Get a timezone object from a string representation of a timezone.

    Parameters
    ----------
    timezone_str : str
        String representation of a timezone. Can be either a timezone name

    Returns
    -------
    pytz.timezone
        A timezone object representing the timezone given by timezone_str
    """
    timezone_str = timezone_str.strip()
    timezone_str = timezone_str.replace(" ", "")
    try:
        # Try to get the timezone directly
        return pytz.timezone(timezone_str)
    except pytz.UnknownTimeZoneError:
        # If that fails, try to parse as a UTC offset
        if timezone_str.startswith("UTC"):
            offset_str = timezone_str[3:]  # remove "UTC" from the start
            try:
                offset_hours = int(offset_str)
                # Create a new timezone representing the offset
                offset = timedelta(hours=offset_hours)
                return pytz.FixedOffset(round(offset.total_seconds() / 60))
            except ValueError:
                return None
        else:
            return None


def prepare_input(full_instance_path):
    """Loads the necessary data from a given path and converts it into usable data formats.

    Args:
        full_instance_path (string): Path to the excel file to read in

    Returns:
        (dict): Various data dictionaries
    """
    # please leave the read in mechanism like this for compatibility with tutOR
    instance_file = open(full_instance_path, "r", encoding="utf8", errors="ignore")

    info_df = pd.read_excel(instance_file.buffer, sheet_name="info", engine="openpyxl")

    info = dict()
    info["w_length"] = info_df.loc[
        info_df["Parameter"] == 'Number of days in a "week"'
    ]["Value"].values[0]
    info["num_weeks"] = info_df.loc[
        info_df["Parameter"] == "Number of weeks of workout schedules"
    ]["Value"].values[0]
    info["preparation_time"] = timedelta(
        minutes=info_df.loc[
            info_df["Parameter"] == "Time for preparation before and after the workout"
        ]["Value"].values[0]
    )
    info["min_workout_time"] = timedelta(
        minutes=info_df.loc[
            info_df["Parameter"] == "Minimum workout time (incl. breaks) in minutes"
        ]["Value"].values[0]
    )
    info["max_workout_time"] = timedelta(
        minutes=info_df.loc[
            info_df["Parameter"] == "Maximum workout time (incl. breaks) in minutes"
        ]["Value"].values[0]
    )
    info["break_time"] = timedelta(
        minutes=info_df.loc[
            info_df["Parameter"] == "Break time between exercises in minutes"
        ]["Value"].values[0]
    )
    info["min_num_workouts_per_week"] = info_df.loc[
        info_df["Parameter"] == "Minimum number of workouts per week"
    ]["Value"].values[0]
    info["max_num_workouts_per_week"] = info_df.loc[
        info_df["Parameter"] == "Maximum number of workouts per week"
    ]["Value"].values[0]
    info["earliest_hour"] = time(
        hour=info_df.loc[info_df["Parameter"] == "Earliest hour to start"][
            "Value"
        ].values[0]
    )
    info["latest_hour"] = time(
        hour=info_df.loc[info_df["Parameter"] == "Latest hour to be finished"][
            "Value"
        ].values[0]
    )
    info["first_day"] = pd.to_datetime(
        info_df.loc[info_df["Parameter"] == "First day in calendar to be optimized"][
            "Value"
        ].values[0],
    ).date()
    info["time_zone"] = get_timezone(
        info_df.loc[info_df["Parameter"] == "Time zone information"]["Value"].values[0]
    )

    exercises_df = pd.read_excel(
        instance_file.buffer, header=1, sheet_name="exercises", engine="openpyxl"
    )
    # remove any unnamed columns
    exercises_df = exercises_df.loc[:, ~exercises_df.columns.str.contains("Unnamed")]
    # body parts are the colums after the first (and only) unnamed column (so a column containing "Unnamed")
    body_parts = list(exercises_df.columns)[7:]
    # get the line where Name is Minimum Weekly Workout Time, and remove it from the dataframe
    min_workout_time = exercises_df.loc[
        exercises_df["Name"] == "Minimum Weekly Workout Time"
    ]
    exercises_df = exercises_df.loc[
        exercises_df["Name"] != "Minimum Weekly Workout Time"
    ]

    exercises = dict()
    # loop through the rows of the exercises dataframe
    for _, row in exercises_df.iterrows():
        if pd.isna(row["Name"]):
            break
        # Name	Category	Sets	Set time in min	Break time between sets in min	Total time	Priority
        name = row["Name"]
        category = row["Category"]
        print(row["Name"], row["Sets"])
        sets = int(row["Sets"])
        set_time = timedelta(minutes=row["Set time in min"])
        break_time = timedelta(minutes=row["Break time between sets in min"])
        total_time = timedelta(minutes=row["Total time"])
        priority = float(row["Priority"])

        # adressed body parts are those columns, where the value is not NaN. If the value is not NaN, it describes the rest time in whole days
        adressed_body_parts = {
            body_part: timedelta(days=row[body_part])
            for body_part in body_parts
            if not pd.isna(row[body_part])
        }

        # add the exercise to the dictionary
        exercises[name] = {
            "name": name,
            "category": category,
            "sets": sets,
            "set_time": set_time,
            "break_time": break_time,
            "total_time": total_time,
            "priority": priority,
            "adressed_body_parts": adressed_body_parts,
        }

    # get the minimum workout time for each body part (0 if NaN)
    minimum_workout_times = {
        body_part: timedelta(minutes=min_workout_time[body_part].values[0])
        if not pd.isna(min_workout_time[body_part].values[0])
        else timedelta(minutes=0)
        for body_part in body_parts
    }

    return info, exercises, minimum_workout_times


def parse_calendar(file_path, tz):
    """
    Parses a calendar file and returns a list of workout events and a list of other events.

    Parameters
    ----------
    file_path : str
        Path to the calendar file
    tz : pytz.timezone
        Timezone of the calendar

    Returns
    -------
    workout_events : list[(datetime, datetime, list[str])]
        List of workout events. Each workout event is a tuple of the start time, the end time, and a list of exercises.
    other_events : list[(datetime, datetime)]
        List of other events. Each other event is a tuple of the start time and the end time.
    """
    # open and parse the calendar
    with open(file_path, "r") as f:
        cal = Calendar.from_ical(f.read())

    workout_events = []
    other_events = []

    for component in cal.walk():
        if component.name == "VEVENT":
            # parse start and end times
            dtstart = component.get("dtstart").dt
            dtend = component.get("dtend").dt
            if isinstance(dtstart, datetime):
                dtstart = dtstart.astimezone(tz)
            if isinstance(dtend, datetime):
                dtend = dtend.astimezone(tz)

            if component.get("summary") == "Workout":
                # parse exercises
                exercises = component.get("description").split("\n")
                workout_events.append((dtstart, dtend, exercises))
            else:
                other_events.append((dtstart, dtend))

    return workout_events, other_events


def add_busy_events(earliest_hour, latest_hour, first_day, number_of_days, timezone):
    """
    Adds busy events during the night to the calendar.

    Parameters
    ----------
    earliest_hour : time
        Earliest hour to be finished
    latest_hour : time
        Latest hour to start
    first_day : datetime.date
        First day in calendar to be optimized
    number_of_days : int
        Number of days to be optimized
    timezone : pytz.timezone
        Timezone of the calendar

    Returns
    -------
    busy_events : list[(datetime, datetime)]
        List of busy events. Each busy event is a tuple of the start time and the end time.
    """
    busy_events = []
    # calculate bedtime start and end for the day before start_day
    bedtime_start = datetime.combine(first_day - timedelta(days=1), latest_hour)
    bedtime_end = datetime.combine(first_day, earliest_hour)
    # add timezone information
    bedtime_start = timezone.localize(bedtime_start)
    bedtime_end = timezone.localize(bedtime_end)
    busy_events.append((bedtime_start, bedtime_end))

    # loop through each day
    for i in range(number_of_days):
        day = first_day + timedelta(days=i)
        # calculate bedtime start and end for this day
        bedtime_start = datetime.combine(day, latest_hour)
        bedtime_end = datetime.combine(day + timedelta(days=1), earliest_hour)
        # add timezone information
        bedtime_start = timezone.localize(bedtime_start)
        bedtime_end = timezone.localize(bedtime_end)
        busy_events.append((bedtime_start, bedtime_end))

    # calculate bedtime start and end for the day after the end
    bedtime_start = datetime.combine(
        first_day + timedelta(days=number_of_days), latest_hour
    )
    bedtime_end = datetime.combine(
        first_day + timedelta(days=number_of_days + 1), earliest_hour
    )
    # add timezone information
    bedtime_start = timezone.localize(bedtime_start)
    bedtime_end = timezone.localize(bedtime_end)
    busy_events.append((bedtime_start, bedtime_end))

    return busy_events


def collapse_events(events, preparation_time, min_workout_time):
    """
    Collapses adjacent or overlapping events into single events.

    Parameters
    ----------
    events : list[(datetime, datetime)]
        List of events. Each event is a tuple of the start time and the end time.
    preparation_time : timedelta
        The maximum amount of time between events for them to be considered adjacent.
    min_workout_time : timedelta
        The minimum amount of time for an event to be considered.

    Returns
    -------
    collapsed_events : list[(datetime, datetime)]
        List of collapsed events. Each event is a tuple of the start time and the end time.
    """
    # sort by start_time
    worklist = list(events)
    changed_any_events = True
    while changed_any_events:
        worklist = list(sorted(worklist, key=lambda x: x[0]))
        changed_any_events = False
        updated_events = []

        was_updated = {event: False for event in worklist}
        for index, event in enumerate(worklist):
            if was_updated[event]:
                continue
            for other_event in worklist[index + 1 : :]:
                if was_updated[other_event]:
                    break

                # Check if the events are overlapping
                if (
                    event[0] <= other_event[0] <= event[1]
                    or event[0] <= other_event[1] <= event[1]
                ):
                    updated_events.append(
                        (min(event[0], other_event[0]), max(event[1], other_event[1]))
                    )
                    was_updated[event] = True
                    was_updated[other_event] = True
                    changed_any_events = True
                    break

                # Check if the events are adjacent
                if (
                    event[1]
                    <= other_event[0]
                    < event[1] + preparation_time * 2 + min_workout_time
                ) or (
                    other_event[1]
                    <= event[0]
                    < other_event[1] + preparation_time * 2 + min_workout_time
                ):
                    updated_events.append(
                        (min(event[0], other_event[0]), max(event[1], other_event[1]))
                    )
                    was_updated[event] = True
                    was_updated[other_event] = True
                    changed_any_events = True
                    break
            if not was_updated[event]:
                updated_events.append(event)
        worklist = updated_events
    return worklist


def max_workout_time_for_day(
    day, calendar, preparation_time, min_workout_time, max_workout_time
):
    """
    Determines the maximum available workouttime for a given day.

    Parameters
    ----------
    day : date
        The day to check
    calenda : list[(datetime, datetime)]
        List of events. Each event is a tuple of the start time and the end time.
    preparation_time : timedelta
        The preparation time before and after a workout.
    min_workout_time : timedelta
        The minimum amount of time for a workout.
    max_workout_time : timedelta
        The maximum amount of time for a workout.

    Returns
    -------
    Optional[Tuple[datetime, timedelta]]
        The maximum available workouttime for the given day, if any.
    """
    # events for the day
    events = [
        event for event in calendar if event[0].date() == day or event[1].date() == day
    ]

    if not events:
        return None

    # sort events by start time
    events = sorted(events, key=lambda x: x[0])

    for index in range(len(events) - 1):
        event_1 = events[index]
        event_2 = events[index + 1]
        assert event_1[0] <= event_2[0]
        assert event_1[1] <= event_2[1]
        assert event_1[1] <= event_2[0]

    # calculate free slots between events
    free_slots = [(events[i][1], events[i + 1][0]) for i in range(len(events) - 1)]

    # subtract preparation time from free slots
    free_slots = [
        (slot[0] + preparation_time, slot[1] - preparation_time) for slot in free_slots
    ]

    # only consider free slots longer than min_workout_time
    free_slots = [slot for slot in free_slots if slot[1] - slot[0] >= min_workout_time]

    if not free_slots:
        # if no suitable free slots, no workout time available
        return None

    # maximum workout time is the longest free slot, but not longer than max_workout_time
    max_duration = timedelta(seconds=0)
    time_start = None
    for slot in free_slots:
        duration = slot[1] - slot[0]
        if duration > max_duration:
            max_duration = duration
            time_start = slot[0] - preparation_time
    assert time_start is not None
    max_workout_time_for_day = min(max_duration, max_workout_time)

    return time_start, max_workout_time_for_day + preparation_time * 2


def solve(full_instance_path, calendar_path):
    """Solving function, takes an instance file, builds and solves a gurobi model and returns solution.

    Args:
        full_instance_path (string): Path to the excel file to read in
        calendar_path (string): Path to the calendar file to read in

    Returns:
         model (gurobipy.model): Solved model instance
    """

    # get data from excel file
    info, exercises, minimum_workout_times = prepare_input(
        full_instance_path=full_instance_path
    )

    # define set of weeks and days as ranges from info
    days = range(1, info["w_length"] * info["num_weeks"] + 1)
    datetime_days = [info["first_day"] + timedelta(days=day - 1) for day in days]

    past_workout_events, other_events = parse_calendar(calendar_path, info["time_zone"])

    # remove any events (from other events) that are not in datetime_days
    other_events = [
        event
        for event in other_events
        if event[0].date() in datetime_days or event[1].date() in datetime_days
    ]

    # to forbid workouts during the night, add dummy events
    other_events += add_busy_events(
        earliest_hour=info["earliest_hour"],
        latest_hour=info["latest_hour"],
        first_day=info["first_day"],
        number_of_days=info["w_length"] * info["num_weeks"],
        timezone=info["time_zone"],
    )

    # collapse adjacent and overlapping events (also collapse if the time between events is too little for a full workout)
    other_events = collapse_events(
        events=other_events,
        preparation_time=info["preparation_time"],
        min_workout_time=info["min_workout_time"],
    )

    # determine maximum available workout time for each day (None if no workout time available)
    workout_times = {
        day: max_workout_time_for_day(
            day=datetime_days[day - 1],
            calendar=other_events,
            preparation_time=info["preparation_time"],
            min_workout_time=info["min_workout_time"],
            max_workout_time=info["max_workout_time"],
        )
        for day in days
    }
    start_workout_times = {
        k: None if v is None else v[0] for k, v in workout_times.items()
    }
    available_workout_times = {
        k: None if v is None else v[1] for k, v in workout_times.items()
    }

    # create gurobipy model
    model = Model("trainmORe")

    # Variable Definition
    x = dict()  # exercise on day
    for exercise in exercises:
        for day in days:
            x[exercise, day] = model.addVar(
                name=f"x_{exercise}_{day}",
                vtype=GRB.BINARY,
            )

    y = dict()  # workout on day
    for day in days:
        y[day] = model.addVar(
            name=f"y_{day}",
            vtype=GRB.BINARY,
        )

    r = dict()  # rest day for body part
    for day in days:
        for body_part in minimum_workout_times:
            r[body_part, day] = model.addVar(
                name=f"r_{body_part}_{day}",
                vtype=GRB.BINARY,
            )

    # Objective (maximize sum of priorities)
    model.setObjective(
        quicksum(
            x[exercise, day] * properties["priority"]
            for exercise, properties in exercises.items()
            for day in days
        ),
        GRB.MAXIMIZE,
    )

    # Constraints
    # link x and y
    for day in days:
        for exercise in exercises:
            model.addConstr(x[exercise, day] <= y[day])

    # link x and r (if d has exercise e with bodypart bp, then the following days are rest days for bp)
    for day in days:
        for exercise, properties in exercises.items():
            adressed_body_parts = properties["adressed_body_parts"]
            for body_part, rest_time in adressed_body_parts.items():
                for i in range(1, rest_time.days + 1):
                    d_prime = day + i
                    if d_prime <= max(days):
                        model.addConstr(x[exercise, day] <= r[body_part, d_prime])

    # link x and r (no exercise of bp on rest day of bp)
    for day in days:
        for exercise, properties in exercises.items():
            adressed_body_parts = properties["adressed_body_parts"]
            for body_part in adressed_body_parts:
                model.addConstr(x[exercise, day] <= 1 - r[body_part, day])

    # cannot workout on a day if no timeslot is available (includes duration check and respects first and last hour)
    for day in days:
        if available_workout_times[day] is None:
            for exercise in exercises:
                model.addConstr(x[exercise, day] == 0)
            model.addConstr(y[day] == 0)

    # Plan a workout such that the maximum available workout time is respected
    for day in days:
        available_time = available_workout_times[day]
        if available_time is not None:
            total_time = (
                quicksum(
                    x[exercise, day] * properties["total_time"].seconds
                    for exercise, properties in exercises.items()
                )
                + 2 * (info["preparation_time"].seconds)
                + (quicksum(x[exercise, day] for exercise in exercises) - 1)
                * info["break_time"].seconds
            )
            model.addConstr(total_time <= available_time.seconds)
            model.addConstr(
                (
                    info["min_workout_time"].seconds
                    + 2 * (info["preparation_time"].seconds)
                )
                * y[day]
                <= total_time
            )

    # each workout contains exactly one warmup exercise
    for day in days:
        if available_workout_times[day] is not None:
            model.addConstr(
                quicksum(
                    x[exercise, day]
                    for exercise, property in exercises.items()
                    if property["category"] == "warm up"
                )
                == y[day]
            )

    # number of workouts per week
    for week in range(info["num_weeks"]):
        model.addConstr(
            quicksum(
                y[week * info["w_length"] + week_day]
                for week_day in range(1, info["w_length"] + 1)
            )
            <= info["max_num_workouts_per_week"]
        )
        model.addConstr(
            quicksum(
                y[week * info["w_length"] + week_day]
                for week_day in range(1, info["w_length"] + 1)
            )
            >= info["min_num_workouts_per_week"]
        )

    # TODO: Minimum Training Time per Body Part per Week
    for week in range(info["num_weeks"]):
        for body_part, minimal_time in minimum_workout_times.items():
            model.addConstr(
                quicksum(
                    x[exercise, week * info["w_length"] + week_day]
                    * properties["total_time"].seconds
                    # * properties["sets"] * properties["set_time"].seconds # see https://moodle.rwth-aachen.de/mod/forum/discuss.php?d=200339#p320118
                    for exercise, properties in exercises.items()
                    for week_day in range(1, info["w_length"] + 1)
                    if body_part in properties["adressed_body_parts"]
                )
                >= minimal_time.seconds
            )

    # TODO: Body Part Resting Days with historic data
    for _, end_time, trained_exercises in past_workout_events:
        date = end_time.date()
        days_until_start = (info["first_day"] - date).days
        assert days_until_start >= 0
        for exercise in trained_exercises:
            if exercise not in exercises:
                # print(exercise)
                continue
            properties = exercises[exercise]
            adressed_body_parts = properties["adressed_body_parts"]
            for body_part, rest_time in adressed_body_parts.items():
                for i in range(1, rest_time.days + 1):
                    d_prime = (
                        1 - days_until_start
                    ) + i  # we have to add 1, since the first day is day 1.
                    if min(days) <= d_prime <= max(days):
                        model.addConstr(r[body_part, d_prime] == 1)

    # update model and solve it
    model.update()
    # model.write("model.lp")
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # save a calendar with optimal workout schedule
        output(
            x,
            start_workout_times,
            {e: v["total_time"] for e, v in exercises.items()},
            info,
        )
    else:
        print("Instance is infeasible.")

    return model


def output(x, start_times, exercise_times, info):
    """Converts MIP solution to importable calendar file. Requires possible workout times for each day as well as
    exercise times and information on the first day of the optimized time frame and break and preparation times.

    Args:
        x (dict): Dictionary with solutions to the x variables of the solved instance
        start_times (dict): Workout start times for each day (keys)
        exercise_times (dict): Time required for performing each exercise (keys)
        info (dict): Used for information on break time, preparation time as well as first day of optimized time frame
    """

    # create a calendar instance
    cal = Calendar()
    cal.add("prodid", "-//workout calendar//")
    cal.add("version", "2.0")

    def create_event(start, end, name, description):
        """Takes information on times and description of an event and creates it.

        Args:
            start (datetime): Date and time of start of the event
            end (datetime): Date and time of end of the event
            name (string): Name of the event
            description (string): Description of the event
        """
        event = Event()
        event.add("summary", name)
        event.add("description", description)
        event.add("dtstart", start)
        event.add("dtend", end)
        cal.add_component(event)
        pass

    def get_duration_and_description(x, exercises, day):
        duration = timedelta(seconds=0)

        count = 0
        names = list()
        for exercise in exercises:
            if x[exercise, day].x > 0.5:
                duration += exercise_times[exercise]
                count += 1
                names.append(exercise)

        if count > 0:
            duration += info["break_time"] * (count - 1)
            duration += info["preparation_time"] * 2

        return duration, "\n".join(names)

    def get_workout_times(duration, day):
        """Calculates the start and end time of the workout.

        Args:
            duration (timedelta): Workout time
            day (datetime): Day in optimization time frame

        Returns:
             start_time (datetime): Date and time of start of the workout
             end_time (datetime): Date and time of end of the workout
        """
        start_time = start_times[day]
        end_time = start_time + duration
        # convert to UTC
        start_time = start_time.astimezone(pytz.utc)
        end_time = end_time.astimezone(pytz.utc)
        return start_time, end_time

    for day in start_times.keys():
        duration, description = get_duration_and_description(x, exercise_times, day)
        if duration <= timedelta(seconds=0):
            continue
        start, end = get_workout_times(duration, day)
        create_event(start, end, "Workout", description)

    # save calendar file to path
    # f = open("workouts.ics", "wb") # maybe this will fix the tutor error message, that only read access is allowed
    # f.write(cal.to_ical())
    # f.close()
    pass


if __name__ == "__main__":
    model_solved = solve(full_instance_path="ex1.xlsx", calendar_path="cal1.ics")
