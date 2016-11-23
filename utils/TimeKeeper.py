"""
Contains classes used to keep track of the running time of the system



Naming conventions used:
module_name, package_name, ClassName, method_name, ExceptionName, function_name,
GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name,
local_var_name

"""
from datetime import datetime

class TimeKeeper(object):
    """Keeps track of the running time of various components in the system"""
    initial_time = None
    last_time = None
    running = False
    timing_sections = []
    current_section = None
    format = None

    def __init__(self, configurationInfo):
        """ Creates a new instance of the time keeper, using any provided
            configuration data.
        """
        self.format = configurationInfo['display_format']
        self.initial_time = datetime.now()

    def start(self, name):
        """ Starts a new timing section with the specified name """
        # Check to see if we have just started executing
        if self.current_section is None:
            self.last_time = datetime.now()
            self.current_section = name
            self.running = True
        else:
            # Add a new timing section, so start by saving the old one
            current_time = datetime.now()
            time_info = [current_time,
                         current_time - self.initial_time,     # Overall elapsed time
                         current_time - self.last_time,        # Section time
                         self.current_section]               # Name of the last section
            self.timing_sections.append(time_info)
            # Prepare the next timer run
            self.last_time = current_time
            self.current_section = name
            self.running = True

    def stop(self):
        """ Halts all time-keeping operations """
        self.running = False
        # Save any existing time sections
        if self.current_section is not None:
            current_time = datetime.now()
            time_info = [current_time,
                         current_time - self.initial_time,     # Overall elapsed time
                         current_time - self.last_time,        # Section time
                         self.current_section]               # Name of the last section
            self.timing_sections.append(time_info)

    def generate_report(self):
        """ Returns a string representation of the times """
        output_string = ""
        for record in self.timing_sections:
            output_string += "%s: %s \t Start: %s \t End: %s \t Duration %s \n" % (
                datetime.strftime(record[0], "%H:%M:%S"),
                record[3],
                str(record[1]),
                str(record[1]  + record[2]),
                "%d.%ds" % (record[2].seconds, record[2].microseconds))
        return output_string
