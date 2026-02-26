import numpy as np
from src import _io, _utils, _errors, _image_utils
# REPLACE _ERRORS WITH HYPERSPY.utils.exceptions!!!
from hyperspy.signals import Signal2D
from matplotlib import pyplot as plt
import warnings

class Images:
    """A class containing the images from particle analysis.
    The images are a HyperSpy signal, which means that the images can 
    be plotted by using the integrated the .plot() function.

    Parameters
    ---------
    path 
        string path to where the images are located (i.e. the directory where the image folders 
        are located == folder where the overview image is stored by default. Typical name is Sutb__
    read_order
        List of particle label names which is also the particle order. Example: ['Stub1-1-1', 'Stub1-1-2', ...]
        Which indicates Stub-ID, Image-ID and particle-ID corresponding to the Image-ID
    dtype
        Data type to set the images to (default: as-read from using pyplot.imread --> float)
    """
    
    def __init__(self, path, label_names, dtype = None):
        
        im, imgs, p_imgs = _io._get_images_from_particle_analysis(path = path, 
                                                                  read_order = label_names,
                                                                  get_individual_particle_images = True,
                                                                  set_dtype = dtype)
        
        self.particle_images = Signal2D(p_imgs)

        self.navigation_signal = Signal2D(imgs)

        self.pa_overview_image = Signal2D(im)

        self.navigation_shape = (len(imgs),)

        self.signal_shape = np.shape(imgs)[1:]

        self.shape = self.navigation_shape + self.signal_shape 

        self.particle_image_shape = np.shape(p_imgs)[1:]

        self.unique_particle_labels = np.arange(1, 1 + len(p_imgs))

        self.phase_map = dict()

        if len(self.shape) == 3: self.is_gridified = False

        else: self.is_gridified = True

        if self.is_gridified: self.phase_map_shape = (self.navigation_shape[0] * self.signal_shape[0], 
                                                     self.navigation_shape[1] * self.signal_shape[1])
        else: self.phase_map_shape = ()

    def __repr__(self):

        if len(self.navigation_shape) == 3: print_shape = ['', self.navigation_shape[0]]

        else: print_shape = [self.navigation_shape[0], self.navigation_shape[0]]

        string0 =  f"<Navigation images, dimensions: ({print_shape[0]}, {print_shape[1]}| {self.signal_shape[-1]}, {self.signal_shape[0]})>\n<Particle images, dimensions: ({self.particle_images.data.shape[0]} | {self.particle_images.data.shape[-1]}, {self.particle_images.data.shape[-2]})>\n\n"

        print_string = "Stored properties\n"

        strings = []
        ignore_strings = []
        for key in self.__dict__.keys(): 

            # Ignore tuples and shapes etc.
            for kwords in ['is_', 'shape']:

                if kwords in key: ignore_strings.append(key)

            # Ignore certain key words and hidden attributes:
            if key not in ignore_strings and key[0] != '_': strings.append(key)

        strings = np.sort(strings)

        for string, counter in zip(strings, np.arange(len(strings))):

            if counter != len(strings)-1: print_string += f" ├── {string}\n"

            else: print_string += f" └── {string}"

        return string0 + print_string

        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%% PRIVATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def _reshape_particle_images(self, new_shape):
        """Reshape the particle images into new_shape provided the new_shape is larger than the stored shape.
        The particle images will only be padded by the difference between the current and the new shape.
        """
        ndim = len(new_shape)

        if ndim != 2: raise ValueError(f"The provided shape {new_shape} is not compatible with two dimensions.")
        
        pshape = self.particle_image_shape

        new_shape = list(new_shape)

        if new_shape[0] > pshape[0] or new_shape[1] > pshape[1]:

            if new_shape[0] < pshape[0]: new_shape[0] = pshape[0]
            if new_shape[1] < pshape[1]: new_shape[1] = pshape[1]

            new_shape = tuple(new_shape)

            print(f"Reshaping particle images: {pshape} -> {new_shape}")

            diff0, diff1 = new_shape[0] - pshape[0], new_shape[1] - pshape[1]

            # Define pad width:
            if diff0 % 2 == 0: diff0 = (diff0 // 2, diff0 // 2)
            else: diff0 = (diff0 // 2, diff0 // 2 + 1)
            if diff1 % 2 == 0: diff1 = (diff1 // 2, diff1 // 2)
            else: diff1 = (diff1 // 2, diff1 // 2 + 1)

            # Map the padding:
            self.particle_images.map(np.pad, pad_width = (diff0, diff1), mode = 'constant', constant_values = 0)

            self.particle_image_shape = self.particle_images.data.shape[-2:]
        
    
    def _gridify_2Dsignals(self):
        """Gridify all signals within the Images class to have a new navigation_shape"""
            
        for attr in self.__dict__.keys():

            #Identify the Signal2D attributes        
            if (type(self.__dict__[attr]) == Signal2D): 
                
                # Identify the Signal2D with navigation shape    
                if np.prod(self.__dict__[attr].data.shape) == np.prod(self.shape):

                    # Identify the signals to gridify:
                    if self.__dict__[attr].data.shape != self.shape:

                        set_signal = self.__dict__[attr].data
                        
                        # Reset the gridified signals:
                        if len(set_signal.shape) == 4:
    
                            set_signal = _image_utils._gridify_4D_array_to_3D(set_signal)
                        
                        self.__dict__[attr] = Signal2D(
                                _image_utils._gridify_3D_array_to_4D(set_signal, 
                                                        to_shape = self.shape, 
                                                        flip_axis = 1))

        self.update_phase_map_shape()

    def _degrifify_2Dsignals(self):
        """Degridify all signals within the Images class to have a new navigation_shape"""
            
        for attr in self.__dict__.keys():

            #Identify the Signal2D attributes        
            if (type(self.__dict__[attr]) == Signal2D): 
                
                # Identify the Signal2D with navigation shape    
                if np.prod(self.__dict__[attr].data.shape) == np.prod(self.shape):

                    # Identify the signals to gridify:
                    if self.__dict__[attr].data.shape != self.shape:
                        
                        self.__dict__[attr] = Signal2D(_image_utils._gridify_4D_array_to_3D(self.__dict__[attr].data))

        self.update_phase_map_shape()

    def _create_particle_mask(self,
                              pimage : np.ndarray,
                              conditions : dict,
                              #remove_edge_artefacts : bool = False,
                              return_single_particle_only : bool = True,
                              background_label : int = 0
                             ):
        """The function attempts to make a mask of the particle specified by the 
        particle_label argument based on the provided conditions (dictionary).
        If the mask is empty, a warning will be written.

        Parameters
        ----------
        pimage
            Particle image to segment
        conditions
            dictionary of conditions to create the particle mask.
            Note that if multiple threhsolding conditions are employed, each condition
            will be applied separately with the remaining ones. 

            Allowed conditions:
            "<" : smaller than an intensity value
            "<=" : smaller than, or equal to an intensity value 
            ">" : larger than an intensity value
            ">=" : larger than, or equal to an intensity value
            'min_size' : minimum particle size (pixels)
            'max_size' : maximum particle size (pixels)
            'min_diameter' : minimum particle diameter (pixels; axis minor length)
            'max_diameter' : maximum particle diameter (pixels; axis major length)
            'binary_erosion' : erode particles' edges a number of times
            'binary_dilation' : dilate particles' edges a number of times
            'fill_holes' : fill particles' holes
            
        return_single_particle_only
            Whether to only create a phase map/phase mask from the provided conditions
            and return the image's largest particle.

        Example
        -------
        >>> particle_mask = _create_particle_mask(
                particle_label = 10,
                conditions = {'>' : 80, # Particle should have intensity > 80
                              '<=' : 160, # article should have intensity <160 
                              'min_size' : 20, # Minimum particle size: 20, i.e. particles with 20 or less pixels will get removed.
                              'binary_erosion' : 2 # The mask will be eroded, iterating twice
                              })
        >>> particle_mask
        array([[1., 1., 1., ..., 0., 0., 0.],
               [1., 1., 1., ..., 0., 0., 0.],
               [1., 1., 1., ..., 2., 2., .],
               ...,
               [0., 2., 2., ..., 2., 2., 2.],
               [0., 2., 2., ..., 0., 0., 0.],
               [0., 0., 0., ..., 0., 0., 0.]])
        """
        
        first_set_of_conditions = ['<','<=','>','>=']

        if not hasattr(self, 'particle_images'):
            raise AttributeError(f"The class doesn't have any images of the individual particles. See the particle_analysis class' *.load_images() function.")

        thresholding = set(first_set_of_conditions) & set(conditions.keys())
        
        num_intensity_ptypes = len(thresholding)
            
        pIm = pimage.copy()

        if num_intensity_ptypes > 1:

            from copy import deepcopy

            # Create one mask per threshold conditions:
            pmask = np.zeros((num_intensity_ptypes,) + pIm.shape)

            for cond, idx in zip(thresholding, np.arange(num_intensity_ptypes)):

                _conditions = deepcopy(conditions)

                for _cond in thresholding:

                    # Make unique thresholding conditions:
                    if _cond != cond: del _conditions[_cond]

                # Get thresholded masks
                pmask[idx] = _image_utils._get_particle_mask(
                    pimage = pIm,
                    conditions = _conditions,
                    #remove_edge_artefacts = remove_edge_artefacts
                ) * (idx + 1)

            # Remove overlapping conditions:
            overlaps = np.prod(pmask, axis = 0) > 0

            pmask *= ~overlaps

            pmask = np.sum(pmask, axis = 0)

        else:

            pmask = _image_utils._get_particle_mask(
                pIm,
                conditions,
                #remove_edge_artefacts
            )

        if return_single_particle_only:

            from skimage.morphology import label

            _mask = np.zeros_like(pmask)

            pmask = label(pmask > 0)

            unique_labels = np.unique(pmask)
            
            unique_labels = np.delete(unique_labels, np.where(unique_labels == background_label))

            if len(unique_labels) > 1:

                sum_labels = np.zeros((len(unique_labels)))
                
                for u in range(len(unique_labels)):

                    sum_labels[u] = np.sum(pmask == unique_labels[u])

                lab = unique_labels[np.argmax(sum_labels)]
    
                _mask[np.where(pmask == lab)] = 1
                
                pmask = _mask

        return pmask
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%% PUBLIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    def update_phase_map_shape(self):
        """Update the attribute phase_map_shape"""
        if self.is_gridified: self.phase_map_shape = (self.navigation_shape[0] * self.signal_shape[0], 
                                                     self.navigation_shape[1] * self.signal_shape[1])
        else: self.phase_map_shape = ()

    def get_depadded_particle_image(self, particle_label_id):
        """As the particle images have padded (zero-valued) regions so that they have the same shape,
        the current function is intended to remove the padded edges and return only the particle image.

        Parameters
        ----------
        particle_id
            Particle label in the particle_map signal. I.e. particle_id is located where the label is stored
            in the attribute unique_particle_labels

        Returns
        -------
        depadded_particle_image 
            particle image with no pad
        """

        if not hasattr(self, 'particle_images'): 
            
            raise AttributeError("No particle images have been loaded. See the function *.load_images()")

        else:

            if particle_label_id not in self.unique_particle_labels: 
                
                raise ValueError(f"The particle label {particle_label_id} does not exists.")
            
            particle_image = self.particle_images.inav[np.where(self.unique_particle_labels == particle_label_id)[0]].data.squeeze()

            return _image_utils._crop_away_empty_edges(particle_image)

    def get_group_of_depadded_particle_images(self, list_of_particle_labels):
        """Get a list of depadded particle images according to the labels in the argument

        Parameters
        ----------
        list_of_particle_labels
            List of particle labels to extract particle images from
        """

        for p in list_of_particle_labels:

            if p not in self.unique_particle_labels: raise ValueError(f"The label {p} does not exist.")
        
        list_of_images = []

        for p in list_of_particle_labels:

            list_of_images.append(self.get_depadded_particle_image(p))

        return list_of_images

    def create_stitched_image_grid_mask(self, edge_width = 1):
        """Create a grid mask that can be overlayed on a stitched image to highlight the 
        image edges"""

        if self.is_gridified:

            shape = (np.prod((self.navigation_shape)),) + self.signal_shape

        else: shape = self.shape

        mask = np.zeros(shape, bool)

        mask[:, edge_width:-edge_width, edge_width:-edge_width] = True

        mask = _image_utils._stitch_images(mask, self.navigation_shape)

        return mask

    def get_number_of_particles_per_image(self):
        """Return the particle number density per image"""

        correct_nav_shape = False
        
        if len(self.navigation_shape) == 1:

            old_shape = self.navigation_shape

            self.navigation_shape = self.navigation_shape + (1,)

            correct_nav_shape = True
        
        number_map = np.zeros(self.navigation_shape)

        shape = number_map.shape

        for i in range(shape[0]):

            for j in range(shape[1]):

                # Minus background
                number_map[i,j] = len(np.unique(self.particle_map.data[i,j])) - 1 

        if correct_nav_shape:

            self.navigation_shape = old_shape
        
        return number_map

    def map_cropped_particles(self, edge_width = 1):
        """The function iterates through all the images according to its navigation shape
        and looks for particles that might be the same particle, but in different SEM/
        particle_map images. The identified particles' indices will be stored in the object's
        instance Images.cropped_particles_map (hyperspy signal)

        OBS! The function will separate cropped particles by their classes. I.e. it can be 
        an idea to classify the particles before mapping the cropped particles.

        Parameters
        ----------
        edge_width
            Image edge width in number of pixels to look for cropped particles. By default: 1, but a 
            higher value can be of advantage if the SEM images has a high spatial resolution. 
            
        """
        if not hasattr(self, 'particle_map'): raise AttributeError("The class doesn't have the Images attribute 'particle_map'. See the *.identify_particle_regions() function.")
        
        proceed = True

        # Check if the signal is gridified and whether to proceed the search.
        if not self.is_gridified: 

            ans = input(f"The images of shape {self.Images.shape} is not gridified. Proceed? (y/[n])")

            if ans.upper() == 'N' or ans.upper() == '': proceed = False

        if proceed:
        
            cropped_particles_map = _image_utils._get_edge_particles_gridMap_from_labelled_particles_gridMap(
                self.particle_map.data,
                edge_width = edge_width
            )

            print("Cropped particle map -> *.Images.")
            self.cropped_particles_map = Signal2D(cropped_particles_map.copy())

            #unique_particle_labels = np.unique(cropped_particles_map)
    
    def concatenate_two_cropped_particle_images(self, particle_label1 : int, particle_label2 : int):
        """Concatenate two particle images together only if they are at compatible image edges. The function 
        will crop out a region from the stored navigation signal with compatible shapes and return a concatenated
        image. 

        Parameters
        ----------
        particle_label1, particle_label2
            Particle labels as stored in the particle_map

        Returns
        -------
        stitched_particle_image
            Concatenated particle images given that the two particle images are not at an image edge.
        """
        #print('NOT FULLY TESTED ON 3D ARRAY OF IMAGES YET')
        loc0, IM_IDX0 = _image_utils._identify_a_particles_position_wrt_SEM_image_edges(particle_label1,
                                                                           self.particle_map.data)
        loc1, IM_IDX1 = _image_utils._identify_a_particles_position_wrt_SEM_image_edges(particle_label2,
                                                                           self.particle_map.data)

        if (IM_IDX0[0] != IM_IDX1[0]) and (IM_IDX0[1] != IM_IDX1[1]):
            raise ValueError(f"SEM images with navigation indices {IM_IDX0} and {IM_IDX1} are not beside each other.")

        # Check if the images are compatible:
        if self.is_gridified:        
            sem_img0 = self.navigation_signal.inav[IM_IDX0[1],IM_IDX0[0]].data
            region_img0 = self.particle_map.inav[IM_IDX0[1],IM_IDX0[0]].data
            sem_img1 = self.navigation_signal.inav[IM_IDX1[1],IM_IDX1[0]].data
            region_img1 = self.particle_map.inav[IM_IDX1[1],IM_IDX1[0]].data
        else: 
            sem_img0 = self.navigation_signal.inav[IM_IDX0[0]].data
            region_img0 = self.particle_map.inav[IM_IDX0[0]].data
            sem_img1 = self.navigation_signal.inav[IM_IDX1[0]].data
            region_img1 = self.particle_map.inav[IM_IDX1[0]].data

        # The particles position:
        min_pos0 = np.min(np.where(region_img0 == particle_label1), axis = -1)
        max_pos0 = np.max(np.where(region_img0 == particle_label1), axis = -1)
        min_pos1 = np.min(np.where(region_img1 == particle_label2), axis = -1)
        max_pos1 = np.max(np.where(region_img1 == particle_label2), axis = -1)

        # Common size:
        shape0 = (max_pos0[0] + 1 - min_pos0[0], max_pos0[1] + 1 - min_pos0[1])
        shape1 = (max_pos1[0] + 1 - min_pos1[0], max_pos1[1] + 1 - min_pos1[1])
        shape = (max([shape0[0], shape1[0]]), max([shape0[1], shape1[1]]))

        # Crop out common sized particle images
        img0 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img0, 
                                                                  region_img0 == particle_label1,
                                                                  get_shape = shape)

        img1 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img1, 
                                                                  region_img1 == particle_label2,
                                                                  get_shape = shape)

        # Concatenate the equally sized images if they are not at an image edge
        """
        if loc0[0] == 0: # Bottom
            if loc1[0] == 1: # Bottom-Top
                return np.concatenate([img1, img0], axis = 0)
            elif loc1[0] == 0: # Bottom-Bottom
                if loc0[1] == 0 and loc1[1] == 1: # Left-Right
                    return np.concatenate([img1, img0], axis = 1)
                elif loc0[1] == 1 and loc1[1] == 0: # Right-left
                    return np.concatenate([img0, img1], axis = 1)
                else: return np.array([])
            else: return np.array([])
        elif loc0[0] == 1: # Top
            if loc1[0] == 0: # Top-bottom
                return np.concatenate([img0, img1], axis = 0)
            elif loc1[0] == 1: # Top-top
                if loc0[1] == 0 and loc1[1] == 1: # Left-Right
                    return np.concatenate([img1, img0], axis = 1)
                elif loc0[1] == 1 and loc1[1] == 0: # Right-left
                    return np.concatenate([img0, img1], axis = 1)
                else: return np.array([])
            else: return np.array([])
        elif loc0[1] == 0: #Left side
            if loc1[1] == 1: # right side
                return np.concatenate([img1, img0], axis = 1)
            else: return np.arary([])
        elif loc0[1] == 1: # Right side
            if loc1[1] == 0: # Right-left side
                return np.concatenate([img0, img1], axis = 1)
            else: return np.array([])
        else: return np.array([])"""

        return _image_utils._concatenate_two_edge_images(img0, img1, loc0, loc1)
    
    """def _stitch_two_particle_images_old(self, particle_label1, particle_label2, 
                                    shift_threshold : float = 1.0,
                                    remove_non_matched_region : bool = True,
                                    success_threshold : float = 0.25):
        stitch two particle images. The particles are identified by their corresponding labels.

        Parameters
        ----------
        particle_label1, particle_label2
            particle labels (integers)
        shift_threshold
            float describing whether a image stitching is sucessful or not. Default: 1.0 (Eucledian distance)
        remove_non_matched_region 
            Whether to remove the non-matched region from the stitched image. The non-matched region originates
            from the larger area extracted from the particle's corresponding SEM image.
            Default: True
        success_threshold
            A normalised cross-correlation score dictating whether a stitching succeeded or not.
            Default: 0.5

        Returns
        -------
        stitched_image
            numpy.ndarray of the stitched image

        (success, score)
            Whether the stitching was successful or not based on image shifting (bool)
            and the corresponding ncc score between the overlapping region of the shifted 
            image and the statinary image 
        
        
        num_particles = self.particle_images.axes_manager[0].size

        if particle_label1 > num_particles:

            raise ValueError(f"The particle label argument 1 ({particle_label1}) is higher than the number of particles ({num_particles}).")
        
        if particle_label2 > num_particles:

            raise ValueError(f"The particle label argument 2 ({particle_label2}) is higher than the number of particles ({num_particles}).")
        
        # Get the two particle images
        img0 = self.get_depadded_particle_image(particle_label1)
        img1 = self.get_depadded_particle_image(particle_label2)

        # Identify the particles' position wrt. the SEM image edges:
        loc0, IM_IDX0 = _image_utils._identify_a_particles_position_wrt_SEM_image_edges(particle_label1, self.particle_map.data)
        loc1, IM_IDX1 = _image_utils._identify_a_particles_position_wrt_SEM_image_edges(particle_label2, self.particle_map.data)

        # Pad the images to make them have identical shapes:
        pim0, pim1 = _image_utils._copy_images_into_common_array_shape(img0, img1)

        # Identify the SEM and particle images
        sem_img0 = self.navigation_signal.inav[IM_IDX0[1],IM_IDX0[0]].data
        region_img0 = self.particle_map.inav[IM_IDX0[1],IM_IDX0[0]].data
        sem_img1 = self.navigation_signal.inav[IM_IDX1[1],IM_IDX1[0]].data
        region_img1 = self.particle_map.inav[IM_IDX1[1],IM_IDX1[0]].data

        # Crop out a larger region from the SEM image. Have this region to be a scalar if it's not too lage
        if pim0.shape[0] / sem_img0.shape[0] < 0.1 and pim0.shape[1] / sem_img0.shape[1] < 0.1: 
            im0, extra_pixels = _image_utils._crop_out_a_random_size_larger_particle_area_from_SEM_image(sem_img0, region_img0, particle_label1, pim0.shape, loc0)
        elif pim0.shape[0] / sem_img0.shape[0] < 0.1:
            im0 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img0, region_img0 == particle_label1, get_shape = (pim0.shape[0], int(0.1 * sem_img0.shape[0])))
        elif pim0.shape[1] / sem_img0.shape[1] < 0.1:
            im0 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img0, region_img0 == particle_label1, get_shape = (int(0.1 * sem_img0.shape[1]), pim0.shape[1]))
        else:
            # The image is either large or long - crop out a similarly shaped image
            im0 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img0, region_img0 == particle_label1, get_shape = pim0.shape)
        
        # Get the same shape for im1 as im0:
        im1 = _image_utils._crop_out_a_specified_particle_area_from_SEM_image(sem_img1, region_img1 == particle_label2, get_shape = im0.shape)

        # Pad the images --> avoid strange artefacts in the stitching
        pad = np.max(im0.shape)
        img0_pad = np.pad(im0.astype(float).copy(), pad_width = pad)
        img1_pad = np.pad(im1.astype(float).copy(), pad_width = pad)

        shifted_image, shift = phase_cross_correlate_images(img0_pad, img1_pad)
        success = _image_utils._assess_successful_stitching(shift, shift_threshold = shift_threshold)
        
        if success: # If shifting succeeded or not

            from scipy.ndimage import binary_fill_holes as bfh
            
            # Sucessful stitching
            mask_shift = _image_utils._get_rectangle_mask_after_image_shift(shifted_image, top_hat_filter = True)
            mask_im = _image_utils._get_rectangle_mask_after_image_shift(img1_pad)
            overlap_mask = mask_shift * mask_im
            sum_mask = (mask_shift.astype(int) + mask_im.astype(int)) > 0

            diff = (bfh(sum_mask).astype(int) - sum_mask.astype(int)).astype(bool)
            sum_mask[diff] = True
            mask_im[diff] = True

            # Create a rectangular overlap mask to assess the shifted image and the
            # statinary image's similarity - metric is normalised cross-correlation.
            rect_mask = _image_utils._rectangulerize_mask(overlap_mask)

            if rect_mask.sum() > 0:
                
                overlap_score = _utils.ncc(_image_utils._remove_empty_edges_from_image(img1_pad*overlap_mask),
                                           _image_utils._remove_empty_edges_from_image(shifted_image*overlap_mask))
                
                rect_score = _utils.ncc(_image_utils._remove_empty_edges_from_image(img1_pad*rect_mask),
                                        _image_utils._remove_empty_edges_from_image(shifted_image*rect_mask))
                
                score = overlap_score * rect_score
            
            else: score = 0

            if score >= success_threshold:

                # Replace everythin but the shift region with the original image
                stitched_image = shifted_image.copy()
                stitched_image[mask_im] = img1_pad[mask_im]
                stitched_image *= sum_mask # Remove shift artefacts
    
                # Remove stripes and other shifting artefacts
                #stitched_image *= (stitched_image > thr(stitched_image))

                # If only keep the original particle images in the stitched image:
                if remove_non_matched_region: 
                    
                    # Identify the original particle images in the stitched image
                    p_mask = np.zeros_like(stitched_image, bool)
                    for pim in [img0, img1]: p_mask += _image_utils._get_single_particle_map(stitched_image, pim)
        
                    # Mask out the original images:
                    masked_stitched_image = (p_mask * stitched_image)
    
                else: masked_stitched_image = stitched_image
    
                # Remove pads
                return _image_utils._remove_empty_edges_from_image(masked_stitched_image), True

            else: return np.array([]), False 
            
        else: return np.array([]), success"""
    
    def stitch_two_particle_images(self, 
                                   particle_label1 : int, 
                                   particle_label2 : int, 
                                   shift_threshold : float = 1.0,
                                   remove_non_matched_region : bool = False,
                                   success_threshold : float = 0.25):
        """Try stitching two individual particle images defined by the particle labels (first two 
        arguments).

        Parameters
        ----------
        particle_label1, particle_label2
            particle labels (integers)
        shift_threshold
            float describing whether a image stitching is sucessful or not. Default: 1.0 (Eucledian distance)
        remove_non_matched_region 
            Whether to remove the non-matched region from the stitched image. The non-matched region originates
            from the larger area extracted from the particle's corresponding SEM image.
            Default: True
        success_threshold
            A normalised cross-correlation score dictating whether a stitching succeeded or not. This threshold
            is compared with the product of the NCC score between the overlapping region and a larger area with 
            the stitched image. Default: 0.25

        Returns
        -------
        stitched_image
            depadded numpy.ndarray of the stitched image

        success
            Whether the stitching was successful or not based on image shifting (bool) and ncc
            score between the individual images and the stitched image in the stitched region.
        """
        num_particles = self.particle_images.axes_manager[0].size

        if particle_label1 > num_particles:

            raise ValueError(f"The particle label argument 1 ({particle_label1}) is higher than the number of particles ({num_particles}).")
        
        if particle_label2 > num_particles:

            raise ValueError(f"The particle label argument 2 ({particle_label2}) is higher than the number of particles ({num_particles}).")

        if not hasattr(self, 'particle_images'): 
            
            raise AttributeError(f"The object has no map of the particles.")
        
        # Get the two particle images
        img0 = self.get_depadded_particle_image(particle_label1)
        img1 = self.get_depadded_particle_image(particle_label2)

        img1, success = _image_utils._stitch_two_particle_images(
            img0 = img0, img1 = img1,
            particle_label1 = particle_label1, 
            particle_label2 = particle_label2,
            particle_map = self.particle_map,
            navigation_signal = self.navigation_signal,
            shift_threshold = shift_threshold,
            remove_non_matched_region = remove_non_matched_region,
            success_threshold = success_threshold)

        img1 = img1.astype(self.particle_images.data.dtype)

        return img1, success

    def stitch_group_of_particles(self,
                                  list_of_particles : list | np.ndarray,
                                  shift_threshold : float = 1.0,
                                  remove_non_matched_region : bool = False,
                                  success_threshold : float = 0.25,
                                  looping_progressbar = False,
                                  directly_stitch_edge_particles = False):
        """Loop through the list/array group_of_particles and try to stitch them together into
        one particle image
        
        Parameters
        ----------
        group_of_particles
            list of particle lables to try to stitch together into one.
        looping_progressbar
            Whether to show a looping progressbar or not. False by default.
        
        Returns
        -------
        stitched_image
            stitched particle image (np.ndarray)
        successfully_stitched_particle_labels
            list of successfully stitched particle labels
        unsuccessfully_stitched_particle_labels
            list of unsuccessfully stitched particle labels
        """

        if not hasattr(self, 'particle_map'): 

            raise AttributeError(f"")

        if not hasattr(self, 'navigation_signal'):
            
            raise AttributeError(f"")

        if len(np.shape(list_of_particles)) != 1:

            raise AttributeError(f"The shape of the argument 'list_of_particles' ({np.shape(list_of_particles)}) is not valid. It should be one dimensional.")
        
        stitched_im, success_labels, fail_labels = _image_utils._stitch_group_of_particles(
            group_of_particle_images = self.get_group_of_depadded_particle_images(list_of_particles),
            group_of_particle_labels = list_of_particles,
            particle_map = self.particle_map,
            navigation_signal = self.navigation_signal,
            shift_threshold = shift_threshold,
            remove_non_matched_region = remove_non_matched_region,
            success_threshold = success_threshold,
            looping_progressbar = looping_progressbar,
            directly_stitch_edge_particles = directly_stitch_edge_particles)

        stitched_im = stitched_im.astype(self.particle_images.data.dtype)
        
        return stitched_im, success_labels, fail_labels

    def cluster_group_of_cropped_particles(self,
                                           list_of_cropped_particle_labels : list,
                                           st_im : np.ndarray,
                                           conditions : dict):
        """By mapping individual particle images as defined by the argument list_of_cropped_particles, onto the stitched particle images stored in the unrelabelled_stitched_particle_images attriute, identify which particle images corresponds to which part of the stitched image, and group then these. The individual particle labels and their composition will be updated accordingly.

        Parameters
        ----------
        list_of_cropped_particle_labels
            List with lists of grouped particle labels that are identified as cropped.
        st_im 
            Stitched image (ndarray) which the list of cropped particle labels corresponds to.
        conditions
            A dictionary of conditions to segment particle regions in the stitched particle images.
            See the image_utils' function _get_particle_mask() documentation for how it works.
            
        Returns
        -------
        particle_labels
            List of particle labels according to the image 'Label_image' in new_particle_images 
            (See below)
        particle_images
            Array of particle images corr. to the labels in th eparticle_labels list (in the same 
            order as it)
        new_particle_images
            Dictionary with particle images cropped out from the stitched image. The key arguments
            corresponds to the labels given in particle_labels.
            The labelled stitched image is also stored here (key: 'Label image')
        """
        if not hasattr(self, 'unrelabelled_stitched_particle_images'):
            
            warnings.warn("The object has no record of cropped particle images.")
        
        else:

            from skimage.morphology import label
            from scipy.ndimage import binary_dilation

            # Apply conditions to the stitched image and label it
            masked_stitch = self._create_particle_mask(
                pimage = st_im.copy(), 
                conditions = conditions,
                return_single_particle_only=False
            )

            masked_stitch = label(masked_stitch)

            # Get list of individual particle images
            pimages = self.get_group_of_depadded_particle_images(
                list_of_cropped_particle_labels
            )

            particle_image_masks = []

            for pim_id in range(len(list_of_cropped_particle_labels)):
                particle_image_masks.append(
                    self._create_particle_mask(
                        pimage = pimages[pim_id],
                        conditions = conditions, 
                        return_single_particle_only=False
                    )
                )

            # Map the particle labels based on the segmented stitched image
            particle_labels, particle_image_masks = _image_utils._segment_clustered_particles(
                particle_images = pimages,
                particle_image_masks = particle_image_masks,
                stitched_image = st_im,
                segmented_stitched_image=masked_stitch,
                conditions = conditions
            )

            unique_labels = np.unique(particle_labels)

            new_particle_images = dict()

            for pid in unique_labels:

                new_particle_image = _image_utils._rectangulerize_mask(
                    binary_dilation(masked_stitch == pid, 
                    iterations = 2)
                ) * st_im

                new_particle_images[int(pid)] = _image_utils._remove_empty_edges_from_image(
                    new_particle_image
                )

            new_particle_images['Label_image'] = masked_stitch

            return particle_labels, particle_image_masks, new_particle_images
    
    
    def get_entire_phase_map(self, unique_classes, set_background_as_label = 'Background'):
        """Return a numpy array of all the phase maps as a stitched image

        Returns
        -------
        phase_map
            The entire phase map as a numpy array. Note that the phase labels (integers) represent
            the different phases in the same order as self.phase_map.keys()

        labels
            The phase labels representing the phases' IDs in the phase map
        """
        if len(self.phase_map) == 0: raise ValueError("The object's phase maps are empty.")
        
        if type(unique_classes) in (list, tuple): unique_classes = np.ndarray(unique_classes)  
        
        from tqdm import tqdm
        
        phase_map = np.zeros(self.phase_map_shape, int)
        
        labels = np.arange(1, len(unique_classes) + 1)
        
        for pm, scale in tqdm(zip(list(self.phase_map.keys()), labels), total = len(labels)):
            
            phase_map[_image_utils._stitch_images(self.phase_map[pm], shape = self.navigation_shape)] = scale

        labels = np.insert(unique_classes, 0, 'Background')
        
        return phase_map, labels

    def plot(self, unique_classes, colours = None, 
             background_colour = 'whitesmoke', set_background_as_label = 'Matrix'):
        """Plot the phase maps stored in Images class"""
        
        from matplotlib import colors
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        auto_colouring  = False

        colour_type = type(colours)
        
        if colours is not None: 

            if len(colours) < len(unique_classes): 

                print('Provided colors do not match the number of classes. Random colours will be generated.')

                auto_colouring = True

            else: 

                if colour_type == dict: 

                    colours = [colours[cl] for cl in unique_classes]

        else: auto_colouring = True

        phase_map, phase_labels = self.get_entire_phase_map(unique_classes, set_background_as_label = 'Matrix')

        phase_vals = np.unique(phase_map)

        # Create a unique color map
        if auto_colouring: 

            if len(unique_classes) < 11: 
                
                print('Colouring according to tableau colors')
                
                colours = [col for col in list(colors.TABLEAU_COLORS.keys())[:len(unique_classes)]]
            
            else: 
                
                print('Generating random colours')
                # Alternatively, use: colors.CSS4_COLORS
                colours = [_utils._generate_random_rgb_color() for i in range(len(unique_classes))]
        
        # Insert background colour: black
        colours.insert(0, colors.to_rgb(background_colour))
            
        cmap = colors.ListedColormap(colours)
        
        norm = colors.BoundaryNorm(np.arange(-0.5, phase_vals.max() + 1.5, 1), cmap.N)

        print(phase_labels, phase_vals)
        # Plotting
        fig, ax = plt.subplots()
        cax = ax.imshow(phase_map, cmap = cmap, norm = norm)
        # Add a colorbar with a label
        cbar = fig.colorbar(cax, ticks = phase_vals)
        cbar.ax.set_yticklabels(phase_labels) 
        plt.axis('off')
        plt.show()
    
def get_stitched_image_from_overlapping_images(img0_part, img1_part, plot_result = False):
    """The function tries to identify overlapping regions between the two images provided in the arguments.

    See https://github.com/hakonanes/correlated-grains-particles-workflow/blob/main/notebooks/particle_detection.ipynb.

    Parameters
    ----------
    img0_part, img1_part
        np.ndarray representing images that will be tried to be stitched

    Returns
    -------
    stitched_image
        np.ndarray representing the stitched image
    shift
        list of shifted values
    """
    import scipy.ndimage as scn
    import skimage.exposure as ske
    import skimage.color as skc
    import skimage.registration as skr
    import skimage.measure as skm
    import skimage.transform as skt
    
    shift, error, diffphase = skr.phase_cross_correlation(
        reference_image=img1_part, moving_image=img0_part, upsample_factor=100)
    
    img0_part_shifted = scn.fourier_shift(np.fft.fftn(img0_part), shift)
    img0_part_shifted = np.fft.ifftn(img0_part_shifted)

    if plot_result:
        if img0_part.shape[0] > 3 * img0_part.shape[1]:
            fig = plt.figure(figsize=(16, 8))
            ax1 = plt.subplot(1, 5, 1)
            ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot(1, 5, 4)
            ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)
        elif img0_part.shape[0] < 3 * img0_part.shape[1]:
            fig = plt.figure(figsize=(16, 8))
            ax1 = plt.subplot(5, 1, 1)
            ax2 = plt.subplot(5, 1, 2, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot(5, 1, 3, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot(5, 1, 4)
            ax5 = plt.subplot(5, 1, 5, sharex=ax1, sharey=ax1)
        else: 
            fig = plt.figure(figsize=(10, 10))
            ax1 = plt.subplot(1, 5, 1)
            ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
            ax3 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
            ax4 = plt.subplot(1, 5, 4)
            ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)
    
        ax1.imshow(img0_part, cmap='gray')
        ax1.set_title('img0')
        ax1.set_axis_off()
        
        ax2.imshow(img1_part, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('img1')
        
        ax3.imshow(img0_part_shifted.real, cmap='gray')
        ax3.set_axis_off()
        ax3.set_title('img0 shifted')
        
        image_product = np.fft.fft2(img1_part) * np.fft.fft2(img0_part_shifted).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        ax4.imshow(cc_image.real)
        ax4.set_axis_off()
        
        img0_shifted = img0_part_shifted.real
        
        ax5.imshow(img0_shifted + img1_part)
        ax5.set_axis_off()
        ax5.set_title('img0 (shifted) + img1')
    
        print(shift)
        print(error)

    tmp = img0_part_shifted.real > 0

    from scipy.ndimage import binary_erosion, binary_dilation
    tmp = binary_dilation(binary_erosion(tmp)).astype(np.uint8)
    
    arg = np.where(tmp > 0)
    yi, yf, xi, xf = arg[0].min()+1, arg[0].max(), arg[1].min()+1, arg[1].max()
    mask = np.zeros_like(tmp, bool)
    mask[yi:yf,xi:xf] = True

    if plot_result:
        fig, ax = plt.subplots(1,3,sharey = True, sharex = True)
        ax[0].imshow(img0_part_shifted.real * mask)
        ax[0].set_title('Shifted image 0')
        ax[1].imshow(img1_part)
        ax[1].set_title('Image 1')
        ax[2].imshow((img0_part_shifted.real * mask) + img1_part)
        ax[2].set_title('Sum image')

    # Rescale the intensity in the shifted image
    from skimage.exposure import rescale_intensity
    shifted_image = rescale_intensity(img0_part_shifted.real, 
                                  out_range=(img0_part.min(), img0_part.max())).astype(img0_part.dtype) * mask

    # Identify overlapping region
    set0y = set(np.where(shifted_image)[0])
    set0x = set(np.where(shifted_image)[1])
    set1y = set(np.where(img1_part)[0])
    set1x = set(np.where(img1_part)[1])
    
    overlap_yi, overlap_yf = min(set0y & set1y), max(set0y & set1y) + 1
    overlap_xi, overlap_xf = min(set0x & set1x), max(set0x & set1x) + 1
    
    stitched_image = img1_part.astype(float)
    stitched_image += shifted_image.astype(float)

    # Stitch the images
    overlap_region = img1_part[overlap_yi:overlap_yf, overlap_xi:overlap_xf].copy()#.astype(float) + shifted_image[overlap_yi:overlap_yf, overlap_xi:overlap_xf].copy().astype(float)) // 2
    
    stitched_image[overlap_yi:overlap_yf, overlap_xi:overlap_xf] = overlap_region
    
    return stitched_image, shift

def phase_cross_correlate_images(arr1, arr2, plot_result = False):
    """ Estimate image shift to make arr1 overlap with arr2. I.e. arr1 will be shifted, and 
    arr2 is statinary.
    """
    if arr1.shape == arr2.shape:
        import scipy.ndimage as scn
        import skimage.registration as skr
        
        pad = np.min(arr1.shape)
        
        img0_part = arr1.copy()#np.pad(arr1, pad_width = pad)
        img1_part = arr2.copy()# np.pad(arr2, pad_width = pad)
        
        shift, error, diffphase = skr.phase_cross_correlation(
            reference_image=img1_part, moving_image=img0_part, upsample_factor=100)
        
        img0_part_shifted = scn.fourier_shift(np.fft.fftn(img0_part), shift)
        img0_part_shifted = np.fft.ifftn(img0_part_shifted)

        img0_shifted = img0_part_shifted.real

        if plot_result:
            if img0_part.shape[0] > 3 * img0_part.shape[1]:
                fig = plt.figure(figsize=(16, 8))
                ax1 = plt.subplot(1, 5, 1)
                ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
                ax3 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
                ax4 = plt.subplot(1, 5, 4)
                ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)
            elif img0_part.shape[0] < 3 * img0_part.shape[1]:
                fig = plt.figure(figsize=(16, 8))
                ax1 = plt.subplot(5, 1, 1)
                ax2 = plt.subplot(5, 1, 2, sharex=ax1, sharey=ax1)
                ax3 = plt.subplot(5, 1, 3, sharex=ax1, sharey=ax1)
                ax4 = plt.subplot(5, 1, 4)
                ax5 = plt.subplot(5, 1, 5, sharex=ax1, sharey=ax1)
            else: 
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(1, 5, 1)
                ax2 = plt.subplot(1, 5, 2, sharex=ax1, sharey=ax1)
                ax3 = plt.subplot(1, 5, 3, sharex=ax1, sharey=ax1)
                ax4 = plt.subplot(1, 5, 4)
                ax5 = plt.subplot(1, 5, 5, sharex=ax1, sharey=ax1)
            
            ax1.imshow(img0_part, cmap='gray')
            ax1.set_title('img0')
            ax1.set_axis_off()
            
            ax2.imshow(img1_part, cmap='gray')
            ax2.set_axis_off()
            ax2.set_title('img1')
            
            ax3.imshow(img0_part_shifted.real, cmap='gray')
            ax3.set_axis_off()
            ax3.set_title('img0 shifted')
            
            image_product = np.fft.fft2(img1_part) * np.fft.fft2(img0_part_shifted).conj()
            cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
            
            ax4.imshow(cc_image.real)
            ax4.set_axis_off()
        
            ax5.imshow(img0_shifted + img1_part)
            ax5.set_axis_off()
            ax5.set_title('img0 (shifted) + img1')
        
        return img0_part_shifted.real, shift
    
    else:
        
        raise _errors.ShapeError(f'The two provided arrays of shapes {arr1.shape} and {arr2.shape} can not be stitched together.')



    


    