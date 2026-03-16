#!/usr/bin/env python3
import requests
from math import atan2
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Waypoint:
    """Represents a navigation waypoint with position"""
    lon: float  # longitude
    lat: float  # latitude

class OsmGlobalPlanner:
    """
    A path planner using OpenStreetMap data through OSRM service
    """
    
    def __init__(self, osrm_url: Optional[str] = None) -> None:
        """
        Initialize the path planner
        
        Args:
            osrm_url: Custom OSRM server URL. If None, uses public OSRM instance
        """
        self.osrm_url = osrm_url or "http://router.project-osrm.org/route/v1/driving/"

    def get_route(self, start: str, end: str) -> List[Waypoint]:
        """
        Get route between two points and convert to waypoints
        
        Args:
            start: Start coordinates as "longitude,latitude" string
            end: End coordinates as "longitude,latitude" string
            
        Returns:
            List of Waypoint objects containing position and heading
            Empty list if route cannot be found or error occurs
        """
        request_url = f"{self.osrm_url}{start};{end}"
        
        params = {
            "overview": "full",
            "geometries": "geojson"
        }
        
        try:
            response = requests.get(request_url, params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching route: {e}")
            return []
        
        waypoints = []
        
        if data.get('routes'):
            route = data['routes'][0]
            coordinates = route['geometry']['coordinates']
            end_lon, end_lat = map(float, end.split(","))
            coordinates.append([end_lon, end_lat])
            
            for i in range(len(coordinates) - 1):
                lon1, lat1 = coordinates[i]
                waypoints.append(Waypoint(
                    lon=lon1,
                    lat=lat1
                ))
        
        if not waypoints:
            print("No valid path found")
        
        return waypoints

def create_planner(osrm_url: Optional[str] = None) -> OsmGlobalPlanner:
    """
    Factory function to create a planner instance
    
    Args:
        osrm_url: Optional custom OSRM server URL
        
    Returns:
        Configured OsmGlobalPlanner instance
    """
    return OsmGlobalPlanner(osrm_url) 