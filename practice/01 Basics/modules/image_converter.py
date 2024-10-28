import numpy as np
import cv2
import imutils
from google.colab.patches import cv2_imshow


class Image2TimeSeries:
    """
    Converter from image to time series using an angle-based method.
        
    Parameters
    ----------
    angle_step : int, optional
        Angle step for finding the contour points, by default 10.
    """
    
    def __init__(self, angle_step: int = 10) -> None:
        self.angle_step = angle_step

    def _img_preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: grayscale, invert, blur, and threshold.
        
        Parameters
        ----------
        img : np.ndarray
            Raw input image.
        
        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted_img = cv2.bitwise_not(gray_img)
        blurred_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)
        _, prep_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
        return prep_img

    def _get_contour(self, img: np.ndarray) -> np.ndarray:
        """
        Find the largest contour in the preprocessed image.
        
        Parameters
        ----------
        img : np.ndarray
            Preprocessed image.
        
        Returns
        -------
        np.ndarray
            Largest object contour.
        """
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea, default=None)
        if largest_contour is None or cv2.contourArea(largest_contour) <= 500:
            raise ValueError("No valid contour found in the image.")
        return largest_contour

    def _get_center(self, contour: np.ndarray) -> tuple[int, int]:
        """
        Compute the center of the object using contour moments.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour.
        
        Returns
        -------
        tuple[int, int]
            Coordinates of the object center.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            raise ValueError("Contour center cannot be calculated.")
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y

    def _get_coordinates_at_angle(self, contour: np.ndarray, center: tuple[int, int], angle: int) -> np.ndarray:
        """
        Find a contour point located at a specific angle relative to the center.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour.
        center : tuple[int, int]
            Coordinates of the object center.
        angle : int
            Angle in degrees.
        
        Returns
        -------
        np.ndarray
            Coordinates of the contour point at the specified angle.
        """
        angles = np.rad2deg(np.arctan2(*(center - contour).T))
        angles = (angles + 450) % 360  # Normalize angles to [0, 360)
        found = np.isclose(angles, angle)
        if np.any(found):
            return contour[found][0]
        idx = np.argmin(np.abs(angles - angle))
        return contour[idx]

    def _get_edge_coordinates(self, contour: np.ndarray, center: tuple[int, int]) -> list[np.ndarray]:
        """
        Find contour points located at regular angular intervals.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour.
        center : tuple[int, int]
            Coordinates of the object center.
        
        Returns
        -------
        list[np.ndarray]
            List of contour points at specified angular intervals.
        """
        return [
            self._get_coordinates_at_angle(contour, center, angle)
            for angle in range(0, 360, self.angle_step)
        ]

    def _img_show(self, img: np.ndarray, contour: np.ndarray, edge_coordinates: list[np.ndarray], center: tuple[int, int]) -> None:
        """
        Visualize the image with contour, center, and rays.
        
        Parameters
        ----------
        img : np.ndarray
            Raw input image.
        contour : np.ndarray
            Object contour.
        edge_coordinates : list[np.ndarray]
            List of contour points.
        center : tuple[int, int]
            Coordinates of the object center.
        """
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 6)
        cv2.circle(img, center, 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for coord in edge_coordinates:
            cv2.line(img, tuple(center), tuple(coord), (255, 0, 255), 4)
        cv2_imshow(imutils.resize(img, width=200))

    def convert(self, img: np.ndarray, is_visualize: bool = False) -> np.ndarray:
        """
        Convert image to time series based on distance from the center.
        
        Parameters
        ----------
        img : np.ndarray
            Input image.
        is_visualize : bool, optional
            Whether to visualize the process, by default False.
        
        Returns
        -------
        np.ndarray
            Time series representation of the image.
        """
        prep_img = self._img_preprocess(img)
        contour = self._get_contour(prep_img)
        center = self._get_center(contour)
        edge_coordinates = self._get_edge_coordinates(contour.squeeze(), center)

        if is_visualize:
            self._img_show(img.copy(), contour, edge_coordinates, center)

        ts = [np.linalg.norm(coord - center) for coord in edge_coordinates]
        return np.array(ts)
