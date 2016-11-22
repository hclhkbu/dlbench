#!/usr/bin/python

import argparse

fname1 = 'aprofile1'
fname2 = 'aprofile512x2_16'
fname = 'profile1_gputrace.log'
fforward = 'forward-time'


def readfile(fname):
    f = open(fname)
    content = f.readlines()
    line1 = content[0]
    function_index = line1.find('Name')
    calls_index = 5 
    statistic = {}
    function_list = []
    for line in content[1:]:
        function = line[function_index:].strip('\n')
        calls = int(line[0:calls_index].strip())
        statistic[function] = calls
        function_list.append(function)
    return statistic, function_list


def diff(s1, f1, s2, f2, print_diff=True, print_same=True):
    if print_diff:
        print 'diff\t\tcalls1\t\tcalls2\t\tname\n'
    if print_same:
        print 'Same\t\tcalls1\t\tcalls2\t\tname\n'
    for f in f1:
        calls1 = s1[f]
        calls2 = s2[f]
        if calls1 != calls2 and print_diff:
            print '%d\t\t%d\t\t%d\t\t%s'% (calls2-calls1, calls1, calls2, f) 
        if print_same and calls1 == calls2:
            print '%d\t\t%d\t\t%d\t\t%s'% (calls2-calls1, calls1, calls2, f) 

def statistic_time(fname, pattern=None):
    f = open(fname)
    content = f.readlines()
    line1 = content[0]
    function_index = line1.find('Name')
    if function_index < 0:
        function_index = 112
    duration_index = line1.find('Duration')
    if duration_index < 0:
        duration_index = 10 
    duration_end = duration_index + len('Duration')
    total_seconds = 0.0
    pattern_time = 0.0
    for line in content[1:]:
        if len(line) < 10:
            continue
        function = line[function_index:].strip('\n')
        duration = line[duration_index:duration_end]
        unit = duration[-2:]
        time = 0.0
        if unit == 'ns':
            time = float(duration[0:-2].strip()) * 0.001 * 0.0001 * 0.001
        elif unit == 'us':
            time = float(duration[0:-2].strip()) * 0.001 * 0.001
        elif unit == 'ms':
            time = float(duration[0:-2].strip()) * 0.001
        else:
            print 'Error! There is a unprocessed time: ', line 
        if pattern:
            if function.find(pattern) >= 0:
                pattern_time += time
        total_seconds += time
    print '%s: '%fname, total_seconds
    if pattern:
        print '%s time: %f' % (pattern, pattern_time)


def find_function(fname, function='fermiPlusCgemmLDS128_batched'):
    f = open(fname)
    content = f.readlines()
    print 'Line\t\tDiff\t\tFunction'
    i = 1
    indexs = []
    prev_i = 0
    group = 0
    groups = []
    onegroup = None
    for line in content:
        i += 1
        if line.find(function) >= 0:
            indexs.append(i)
            diff = i-prev_i
            print '%d\t\t%d\t\t%s' % (i, diff, function)
            prev_i = i
            if diff > 10:
                if onegroup and len(onegroup) <= 1:
                    onegroup.append(i) 
                else:
                    onegroup = []
                    onegroup.append(i)
                groups.append(onegroup)
                group += 1
            elif diff < 10:
                onegroup.append(i)
    print '\nGroups:[%d]'%len(groups)
    for group in groups:
        print group 

    return indexs


def print_index_diff(indexs):
    i0 = indexs[0]
    for i in indexs[1:]:
        print i - i0
        i0 = i

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", help="function name to analyze", action="store_true")
    parser.add_argument("-v", "--verbose", help="print the version of this tools", action="store_true")
    parser.add_argument("-s", "--statistic", help="statistic the total time")
    parser.add_argument("-F", "--subfunction", help="statistic the specified fucntion time", default=None)
    parser.add_argument("-d", "--diff", help="diff two files", action="store_true")
    args = parser.parse_args()
    if args.statistic:
        statistic_time(args.statistic, args.subfunction)
    if args.fname: 
        indexs = find_function(fname, function=args.fname)
    if args.diff:
        s1, f1 = readfile(fname1)
        s2, f2 = readfile(fname2)
        diff(s1, f1, s2, f2, print_diff=False, print_same=True)

if __name__ == '__main__':
    main()
