#!/usr/bin/env python
import array
import csv
from collections import deque
from bisect import bisect_right
import heapq
import math
from time import perf_counter
import random

from collections import namedtuple
from scipy.spatial import cKDTree as KDTree
# KDTree for storing 2 dimensional location in binary tree
def kdtree():

    x_min = 0
    y_min = 0
    x_max = 400
    y_max = 400

    def create_data(n):
        file_ = open('cities1000.csv', 'w')
        with file_:
            writer = csv.writer(file_)
            for i in range(0, n):
                data = [i, random.randint(0, 400), random.randint(0, 400)]

                writer.writerow(data)
            file_.close

    class Root_Node:
        def __init__(self):
            self.child = None
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max


    class Internal_Node:
        def __init__(self):
            self.axis = None    # X or Y axis
            self.axis_val = None
            self.left = None
            self.right = None
            self.parent = None


    class Leaf_Node:
        def __init__(self, points):
            self.parent = None
            self.data = points


    def visualize(root):
        node = root.child
        if node is None:
            print('no points')
            return
        q = deque()
        q.append(node)
        while q:
            size = len(q)
            for _ in range(size):
                curr_node = q.popleft()
                if curr_node.parent.__class__.__name__ == 'Internal_Node':
                    print(curr_node.parent.axis, '=',
                        curr_node.parent.axis_val, ' -> ', end='')
                if curr_node.__class__.__name__ == 'Leaf_Node':
                    # add point[0] if point id is asked
                    points = [[point[1], point[2]] for point in curr_node.data]
                    # to display
                    print(points)
                else:
                    print(curr_node.axis, '=', curr_node.axis_val)
                    if curr_node.left:
                        q.append(curr_node.left)
                    if(curr_node.right):
                        q.append(curr_node.right)
            print('------------------------------------------------------------------------------------------')

        return


    # a recursive function to insert points into KD tree
    def insert(parent, points):
        if len(points) <= alpha:    # base case
            new_node = Leaf_Node(points)
            new_node.parent = parent
            return new_node

        # if code reaches here then it's an internal node
        new_node = Internal_Node()
        new_node.parent = parent
        # find spread and decide axis to split
        x_spread = [max(i) for i in zip(*points)][1] - [min(i)
                                                        for i in zip(*points)][1]
        y_spread = [max(i) for i in zip(*points)][2] - [min(i)
                                                        for i in zip(*points)][2]
        if x_spread >= y_spread:
            new_node.axis = 'X'
            sorted_points = sorted(points, key=lambda x: (x[1], x[2]))
            median_index = (len(sorted_points)-1)//2
            new_node.axis_val = sorted_points[median_index][1]
            coordinate_list = [p[1] for p in sorted_points]
            partition_index = bisect_right(coordinate_list, new_node.axis_val) - 1
        else:
            new_node.axis = 'Y'
            sorted_points = sorted(points, key=lambda x: (x[2], x[1]))
            median_index = (len(sorted_points)-1)//2
            new_node.axis_val = sorted_points[median_index][2]
            coordinate_list = [p[2] for p in sorted_points]
            partition_index = bisect_right(coordinate_list, new_node.axis_val) - 1

        # divide the points into left and right
        left_points = sorted_points[:partition_index+1]
        right_points = sorted_points[partition_index+1:]

        # handling the corner case when all points go to one subtree and alpha < number of points
        if (not left_points) or (not right_points):
            new_node.left = Leaf_Node(left_points)
            new_node.right = Leaf_Node(right_points)
            new_node.left.parent = new_node
            new_node.right.parent = new_node
            return new_node

        new_node.left = insert(new_node, left_points)   # build left subtree
        new_node.right = insert(new_node, right_points)  # build right subtree

        return new_node


    # naive KNN algorithm
    def naive_knn(x, y, K, naive_heap, dataset):
        for point in dataset:
            x2, y2 = point[1], point[2]
            distance = math.sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y))
            if len(naive_heap) >= K:
                if -1 * distance > naive_heap[0][0]:
                    heapq.heappushpop(naive_heap, [-1 * distance, [x2, y2]])
            else:
                heapq.heappush(naive_heap, [-1 * distance, [x2, y2]])


    # KNN query algorithm on KD tree
    def knn(root, x, y, K, estimate_found, max_heap, visited):
        # searching for the best estimate
        if not estimate_found[0]:
            if root.__class__.__name__ == 'Leaf_Node':
                for p in root.data:
                    x2, y2 = p[1], p[2]
                    distance = math.sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y))
                    if len(max_heap) >= K:
                        if -1 * distance > max_heap[0][0]:
                            heapq.heappushpop(max_heap, [-1 * distance, [x2, y2]])
                    else:
                        heapq.heappush(max_heap, [-1 * distance, [x2, y2]])

                estimate_found[0] = True
                visited.add(root)
                return
            else:
                if root.axis == 'X':
                    if x <= root.axis_val:
                        knn(root.left, x, y, K, estimate_found, max_heap, visited)
                    else:
                        knn(root.right, x, y, K, estimate_found, max_heap, visited)
                else:
                    if y <= root.axis_val:
                        knn(root.left, x, y, K, estimate_found, max_heap, visited)
                    else:
                        knn(root.right, x, y, K, estimate_found, max_heap, visited)

        # estimate found, backtrack and search for better estimates
        if root in visited:
            return
        if root.__class__.__name__ == 'Leaf_Node':
            points = root.data
            for p in points:
                x2, y2 = p[1], p[2]
                distance = math.sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y))
                if len(max_heap) >= K:
                    if -1 * distance > max_heap[0][0]:
                        heapq.heappushpop(max_heap, [-1 * distance, [x2, y2]])
                else:
                    heapq.heappush(max_heap, [-1 * distance, [x2, y2]])

            visited.add(root)
            return
        # if this subtree can be pruned avoid this call
        # it's an internal node
        if len(max_heap) >= K:
            path = []
            curr_node = root
            while curr_node.parent.__class__.__name__ != 'Root_Node':
                if curr_node.parent.left == curr_node:
                    path.append(['L', curr_node.parent.axis,
                                curr_node.parent.axis_val])
                else:
                    path.append(['R', curr_node.parent.axis,
                                curr_node.parent.axis_val])
                curr_node = curr_node.parent

            xx1 = curr_node.parent.x_min
            yy1 = curr_node.parent.y_min
            xx2 = curr_node.parent.x_max
            yy2 = curr_node.parent.y_max
            path.reverse()
            for p in path:
                if p[1] == 'X':
                    if p[0] == 'L':
                        xx2 = p[2]
                    else:
                        xx1 = p[2]
                else:
                    if p[0] == 'L':
                        yy2 = p[2]
                    else:
                        yy1 = p[2]

            # calculate minimum distance between the point and the region
            dx = max(xx1-x, 0, x-xx2)
            dy = max(yy1-y, 0, y-yy2)
            distance = math.sqrt(dx*dx + dy*dy)
            if -1 * distance < max_heap[0][0]:
                visited.add(root)
                return

        if root.left not in visited:
            knn(root.left, x, y, K, estimate_found, max_heap, visited)
        if root.right not in visited:
            knn(root.right, x, y, K, estimate_found, max_heap, visited)
        visited.add(root)


    def solve(queries, dataset, root_child):
        execution_time = []  # in microseconds
        for q in queries:
            X, Y, K = q[0], q[1], q[2]
            naive_heap = []
            start_time = perf_counter()
            naive_knn(X, Y, K, naive_heap, dataset)
            end_time = perf_counter()
            print('naive algo answer : ', sorted(naive_heap))
    
            estimate_found = [False]
            max_heap = []   # (distance, [x,y])
            visited = set()
            knn(root_child, X, Y, K, estimate_found, max_heap, visited)

            answer_list = []
            while max_heap:
                answer_list.append(heapq.heappop(max_heap)[1])
            if len(queries) == 1:
                print('query result -> ', answer_list)
        
        return
    alpha = int(input('Enter value of alpha : '))



TAB_SEP = '\t'

GEONAMES_CITIES_FILE = 'cities1000.txt'

CityInfo = namedtuple('CityInfo', 'name asciiname altnames latitude longitude '
                                  'feature_code country_code country')


class GeoNames:

    def __init__(self):
        self.locations = []
        self.city_info = []
        self.kdtree = None
        self.kdtree_per_country = {}
        self.lat_lng_mapping = {}

        # Setup intial mapping and location dicts
        self.initial_setup()

    def readfile(self):
        """Returns lines in 'cities1000.txt' as generator"""
        try:
            with open(GEONAMES_CITIES_FILE,encoding="utf8") as fileobj:
                for line in fileobj:
                    yield line
        except IOError:
            print('Invalid filename')

    def initial_setup(self):
        if not self.locations or not self.geoname_lines:
            for line in self.readfile():
                items = line.split(TAB_SEP)

                # Create CityInfo object and append to list
                self.city_info.append(CityInfo(items[1], items[2], items[3],
                                               float(items[4]), float(items[5]),
                                               items[7], items[8], 'None'))

                latitude, longitude = float(items[4]), float(items[5])
                self.lat_lng_mapping[(latitude, longitude)] = items
                self.locations.append([latitude, longitude])

        # Insert all locations into KDTree
        self.kdtree = KDTree(self.locations)

    def find_locations(self, country):
        """Find all cities(latitude, longitude) for given country"""
        locations = []

        for cityinfo in self.city_info:
            if country == cityinfo.country_code:
                locations.append((cityinfo.latitude, cityinfo.longitude))

        # Need to return empty array for checking False result
        return locations if locations else array.array('l', [])

    def query(self, location, k, country=None):
        """Query KDTree for k nearest cities"""
        if not country:
            return self.kdtree.query(location, k=k)
        else:
            if not self.kdtree_per_country.get(country):
                locations = self.find_locations(country=country)

                if len(locations) == 0:
                    return (None, locations)

                self.kdtree_per_country[country] = KDTree(locations)

            return self.kdtree_per_country[country].query(location, k=k)

# Global GenoNames object reference
geoname = None


def geonames():
    """Hacky way of getting refrence to GeoNames object"""
    global geoname

    if not geoname:
        geoname = GeoNames()

    return geoname


def is_valid_city(keyword, cityinfo):
    for name in [cityinfo.name, cityinfo.asciiname, cityinfo.altnames]:
        if keyword.lower() in name.lower():
            return True


def is_city(cityinfo):
    """Returns True if city is actual city and not district"""
    for feature_code in ['PPL', 'PPLC', 'PPLA']:
        if feature_code == cityinfo.feature_code:
            return True


def find_cities(keyword):
    """Handler for '/v1.0/cities/{name}' endpoint.
    Returns all cities which match keyword"""
    result = []

    for cityinfo in geonames().city_info:
        if is_valid_city(keyword, cityinfo) and is_city(cityinfo):
            result.append(dict(city=cityinfo.name,
                               country_code=cityinfo.country_code))

    return result if result else 'Not found'


def find_city(city):
    """Finds latitude and longitude of city"""
    for cityinfo in geonames().city_info:
        if is_valid_city(city, cityinfo):
            return (cityinfo.latitude, cityinfo.longitude)

    return (None, None)


def find_k_nearest_cities(city, k, country=None):
    """Handler for '/v1.0/nearest_cities/' endpoint.
    Finds k nearest cities using KDTree"""
    result = []

    (latitude, longitude) = find_city(city)

    if not latitude or not longitude:
        return 'City not found!'

    # Insert latitude & longitude of all cities in geonames db to KDTree
    _, indices = geonames().query((latitude, longitude),
                                  k=k,
                                  country=country)

    if len(indices) == 0:
        return 'Invalid parameters. Please check your parameters'

    # For each index return by query find latitude, longitude
    # which maps to the city info.
    for index in indices:
        if not country:
            latitude, longitude = geonames().kdtree.data[index]
        else:
            latitude, longitude = \
                geonames().kdtree_per_country[country].data[index]

        city = geonames().lat_lng_mapping[(latitude, longitude)]

        result.append(dict(city=city[1], country_code=city[8]))
        print(geonames().lat_lng_mapping[(latitude, longitude)])

    return result
