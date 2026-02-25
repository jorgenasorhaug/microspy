# Particle analysis helping functions
# Dependencies: hyperspy sub-module exspy and tabulate

import numpy as np
import pandas as pd
from exspy.material import elements as ELEMENTS
from exspy.material import weight_to_atomic, atomic_to_weight
from hyperspy.signals import Signal2D, Signal1D
from tabulate import tabulate
import warnings, os

from src import _io, read_metadata, _utils, _images, _attribute_classes, _errors, _colouring, _image_utils

# tqdm(..., desc=" outer", position=0):


AVAILABLE_UNITS = [
    ['At %', 'At%', '[At %]','[At%]','At.%','[At.%]','at%', '[at%]','at.%','[at.%]',],
    ['Mass %', 'Mass%','[Mass %]','[Mass%]','mass%','[mass%]','wt.%','Wt.%','[wt.%]','[Wt.%]','[wt%]','[Wt%]']
]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%% PA CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class ParticleAnalysis: 
    """Create an object of the result from analysing 
    particles' chemistry and geometric properties

    Parameters
    ----------
    arg 
        filename of csv file or a pandas DataFrame

    See also
    --------

    Examples
    --------
    """
    
    # Instance properties:
    def __init__(self, arg):

        # Make metadata structure:
        self.metadata = _attribute_classes.metadata(arg)
        
        # Allocate Particles class
        self.Particles = _attribute_classes.Particles(arg)

        self.is_classified = self.Particles.classes != 'Unclassified'
        
        # Number of particles 
        self.number_of_particles = _io.get_number_of_particles(arg)

        self.particles_composition_shape = (len(self.Particles.elements), self.number_of_particles)

    # Print nice information about the object:
    def __repr__(self):
        
        print("OBS! Mutliple stubs has not been tested yet!")

        calibrated = False

        cal_string = ''
        
        read_images = False

        grid_string = ''

        if hasattr(self, 'Images'):

            read_images = True

            if hasattr(self.metadata, 'navigation_unit'):

                calibrated = True

        if read_images:

            grid_string = str(self.Images.navigation_shape)[1:-1] + ' | '

        if calibrated: 

            scan_area = np.prod(self.Images.navigation_shape) * np.prod(self.Images.signal_shape)

            scan_area *= np.square(self.metadata.navigation_scale)

            num_density = self.number_of_particles / scan_area

            area_density = np.sum(self.Particles.particle_geometry['Area [um²]']) / scan_area
            
            decimal_position_n = _utils.first_nonzero_decimal_position(num_density)

            decimal_position_a = _utils.first_nonzero_decimal_position(area_density)

            cal_string = f'\nScan unit: {self.metadata.navigation_unit}\nParticle number density: {round(num_density, decimal_position_n + 2)} 1/{self.metadata.navigation_unit}\u00b2\nParticle area density: {round(100 * area_density, decimal_position_a + 2)} %'
        
        pa_string = f"<Particle analysis, title: {self.metadata.General.project_name}, dimensions: ({grid_string}{self.number_of_particles})>{cal_string}"
        
        return pa_string

    ##############################################
    ############### FUNCTIONS ####################
    ##############################################
    
    ##############################################
    #%%%%%%%%%%%% Private functions %%%%%%%%%%%%%%
        
    def _identify_single_element_particles(self, return_array = False):
        """Identify particles that only contain single elements"""
        arr = np.sum(self.Particles.composition > 0, axis = 0) == 1
        
        print(f"Number of single element particles: {np.sum(arr)}")
        
        if return_array: return arr

    def _update_particles_concentration(self, decimals = 2):
        """Update the particles' element concentration."""
        total = np.sum(self.Particles.composition, axis = 0)
        
        self.Particles.composition *= (100 / total) #percentage

        self.Particles.composition = np.round(self.Particles.composition, decimals = decimals)

        #if hasattr(self.metadata.Particles, 'matrix_composition'):

        #    print('Updating the matrix composition')

        #    matrix_total = np.sum(self.metadata.Particles.matrix_composition)

        #    self.metadata.Particles.matrix_composition *= (100 / matrix_total) #percentage

    def _check_array_length(self, array):
        """Check the length of an input array and whether it is compatible with the stored number of particles in the object.

        Parameters
        ----------
        array
            Array whose length is to be checked

        Returns
            True if len(array) is identical to the number of stored particles.
        """
        if len(array) != self.number_of_particles: 

            print("The provided array does not match the number of particles")
            
            return False
            
        else: return True

    def _check_class_arguments(self, check_classes):
        """Check if class arguments in check_classes are classes stored in the object.
        if all classes in check_classes are found: 
            return True, []
        else: return False, [list of missing classes]
        """
        if type(check_classes) == str: check_classes = [check_classes]
        elif not type(check_classes) == list: raise TypeError(f"Provided array must be a list, and not a {type(check_classes)}")
        unique_classes = self.get_unique_particle_classes()
        missing_classes = []
        for cl in check_classes: 
            if cl not in unique_classes: missing_classes.append(cl)
        if len(missing_classes) == 0: return True, []
        else: return False, missing_classes

    def _get_max_class_label_character(self):
        """Return the maximum number of characters in the employed classes"""
        classes = self.get_unique_particle_classes(return_list=True)
        return np.max([len(x) for x in classes])
        
    def _get_max_string_length_in_list(lst):
        """Return the longest string length in list of strings"""
        return np.max([len(x) for x in lst])

    def _update_set_navigation_scale(self):
        """Update navigation_signal and particle_images' navigation unit and scale"""
        if not hasattr(self, 'Images'): 
            
            raise AttributeError('Images have not been read yet. See *.load_images()')
        
        else: 
            
            self.set_navigation_scale(self.metadata.navigation_scale,
                                        self.metadata.navigation_unit)

    def _update_particle_label_list(self, matrix_label = 0):
        """Update the unique particle labels in the Images class according to the particle_map signal.

        Parameters
        ----------
        matrix_label
            Integer label representing the matrix (i.e. not of interest)
        """
        if not hasattr(self, 'Images'): raise AttributeError("The object has no Images class attribute. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): raise AttributeError("The object's Images class has no attribute particle_map. See the function *.identify_particle_map().")

        unique_labels = np.unique(self.Images.particle_map)
        
        self.Images.unique_particle_labels = np.delete(unique_labels, np.where(unique_labels == matrix_label))

    def _update_particles_composition_shape(self):
        """Update the object's attribute particles_composition_shape
        """
        self.particles_composition_shape = (len(self.Particles.elements), self.number_of_particles)
    
    def _stitch_cropped_particles(self,
                                  stitching_ncc_threshold = 0.25,
                                  extra_pixels = 0.2,
                                  shift_threshold = 1.0,
                                  remove_non_matched_region = False,
                                  directly_stitch_edge_particles = True,
                                  filter_stitched_images_to_correct_for_stitching_process = False,
                                  looping_progressbar = False,
                                  update_particle_properties = True):
        """The function will attempt to stitch the identified cropped particles so that measurement
        corrections can be  done. 
        
        Note that if update_particle_properties if True, the pair of stitched particles' chemical 
        composition will be averaged and their corresponding labels updated in all lists and particle
        maps by keeping the lowest label among the two.

        Parameters
        ----------
        stitching_ncc_threshold
            Float value describing the lowest allowable ncc score between two images' overlapping 
            region before stitching is considered successful.
            Default: 0.5.
        extra_pixels
            Float (percentage) defining how many extra pixels wrt. SEM image widht or height to 
            add to improve the chance of correct stitching. 
        shift_threshold 
            A number that assess whether the image stitching was successful or not by how much it was shifted 
            (Eucledian distance). Default: 1.0
        remove_non_matched_region 
            Whether to remove the region that was initially not part of the stored particle image from the 
            stitched image. By default: True
        directly_stitch_edge_particles 
            If particle stitching fails for some reason, concatenate the particle images directly. 
        filter_stitched_images_to_correct_for_stitching_process
            Whether to smooth the stitched images or not. A good reason to do this is to remove artefacts from the image shifting.
            See skimage.registration's phase_cross_correlation function.
        update_particle_properties
            Wheter to correct the cropped particle pairs' chemical composition and geometry.
        
        Returns
        -------
        Example
        -------
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> s.Images
        Particle analysis imags:
            SEM images: <(16 | 1536,2048) | Particle images: (1000 | (500,400))>

        >>> s.gridify_SEM_images(navigation_shape = (4,4))
        >>> s.Images
        Particle analysis imags:
            SEM images: <(4,4 | 1536,2048) | Particle images: (1000 | (500,400))>

        # Identify the particle regions
        >>> s.identify_particle_regions()
        >>> s.stitch_cropped_particles()
        >>> ...
        """
        stitch = True
        
        if filter_stitched_images_to_correct_for_stitching_process:
            from scipy.ndimage import median_filter
        from scipy.ndimage import binary_fill_holes as bfh
        
        if not hasattr(self, 'Images'): raise AttributeError("No images have been read yet. See the function *.load_images()")
        
        if not hasattr(self.Images, 'cropped_particles_map'): 
            
            ans = input("Cropped particles have not been mapped yet. Search for cropped particles? ([y]/n)")

            if ans.upper() == 'Y' or ans == '': self.identify_cropped_particles()

            else: stitch = False

        if not update_particle_properties:

            ans = input("Correct the stitched particles' properties? ([y]/n)")

            if ans.upper() == 'Y' or ans == '': update_particle_properties = True

        if stitch:
            
            if len(self.metadata.Particles.cropped_particle_clusters) > 0: 

                from skimage.exposure import rescale_intensity
                from tqdm import tqdm
    
                # List of arrays with particle labels
                particle_clusters = self.metadata.Particles.cropped_particle_clusters
                num_clusters = len(particle_clusters)
                stitched_particle_images = []
                successfully_stitched = []
                unsuccessfully_stitched = []
    
                for cl in tqdm(range(num_clusters), position=0, desc = "Stitching particles"):
                    
                    num_particles_in_cluster = len(particle_clusters[cl])
                    
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%% > 2 PARTICLE IMAGES TO BE STITCHED TOGETHER: %%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if num_particles_in_cluster > 2:
    
                        minI = 0
                        maxI = 0
    
                        for pim in particle_clusters[cl]: 
                            temp_im = self.get_depadded_particle_image(pim)
                            minI = min([minI, temp_im.min()])
                            maxI = min([maxI, temp_im.max()])
    
                        # Whether to append the second group of particles as an individual particle or not
                        append_second_stitched_image = not directly_stitch_edge_particles
    
                        img1, temp_success, temporarily_not_stitched = self.Images.stitch_group_of_particles(
                            list_of_particles = particle_clusters[cl],
                            shift_threshold = shift_threshold,
                            remove_non_matched_region = remove_non_matched_region,
                            success_threshold = stitching_ncc_threshold,
                            looping_progressbar = looping_progressbar,
                            directly_stitch_edge_particles = directly_stitch_edge_particles
                        )
    
                        # Stitch the remaining non-stitched particles (plural)
                        if len(temporarily_not_stitched) >= 2:
                                
                            if len(temporarily_not_stitched) == 2:
                                
                                img2_, successful_stitching2_ = self.Images.stitch_two_particle_images(
                                    particle_label1 = temporarily_not_stitched[0],
                                    particle_label2 = temporarily_not_stitched[1],
                                    shift_threshold = shift_threshold,
                                    remove_non_matched_region = remove_non_matched_region,
                                    success_threshold = stitching_ncc_threshold)
    
                                if successful_stitching2_:
                                    
                                    temp_success2_ = temporarily_not_stitched.copy()
    
                            else: 
                                
                                img2_, temp_success2_, temporarily_not_stitched = self.Images.stitch_group_of_particles(
                                    list_of_particles = temporarily_not_stitched,
                                    shift_threshold = shift_threshold,
                                    remove_non_matched_region = remove_non_matched_region,
                                    success_threshold = stitching_ncc_threshold,
                                    looping_progressbar = looping_progressbar,
                                    directly_stitch_edge_particles = directly_stitch_edge_particles)
    
                                successful_stitching2_ = len(temp_success2_) > 0
    
                            if successful_stitching2_ and directly_stitch_edge_particles:
    
                                concatible, (img1_loc_, loc1_), axis = _image_utils._check_two_stitched_particle_images_concatenation_compatibilities(
                                    list_of_labels0 = temp_success,
                                    list_of_labels1 = temp_success2_,
                                    particle_map = self.Images.particle_map.data)
    
                                if concatible:
    
                                    # Make the two images into the same size:
                                    pim1, pim2 = _image_utils._copy_images_into_common_array_shape(img1, img2_)
                                    
                                    # Remove padding along a specific axis:
                                    pim1 = _image_utils._remove_padding_along_axis(pim1, axis = axis)
                                    pim2 = _image_utils._remove_padding_along_axis(pim2, axis = axis)
                                                                    
                                    # Concatenate:
                                    img1 = _image_utils._concatenate_two_edge_images(pim1, pim2, img1_loc_, loc1_)
    
                                    for label in temp_success2_:
                                        temp_success.append(label)
                                    temporarily_not_stitched = []
    
                                else: append_second_stitched_image = True
    
                            if append_second_stitched_image: # img1 will be appended below
    
                                if filter_stitched_images_to_correct_for_stitching_process: 
                                    # Filter to remove artefacts from stitching
                                    img2_ = median_filter(img2_, size = 2)
                        
                                if len(temporarily_not_stitched) > 0:
                                    
                                    unsuccessfully_stitched.append(np.sort(np.array(temporarily_not_stitched)))
            
                                if len(temp_success2_) > 0:
            
                                    successfully_stitched.append(np.sort(np.array(temp_success2_)))
            
                                if np.shape(img2_) != (0,): 
                                    
                                    stitched_particle_images.append(img2_)
        
                        elif (len(temporarily_not_stitched) == 1) and directly_stitch_edge_particles: 
                            # Only one particle image hasn't been stitched with the rest yet
                               
                            img1_ = _image_utils._concatenate_a_stitched_image_with_a_non_stitched_image(
                                temporarily_not_stitched[0],
                                temp_success.copy(),
                                img1.copy(),
                                self.Images.particle_map.data,
                                self.Images.navigation_signal.data,
                                to_shape = img1.shape)
    
                            if len(img1_.shape) > 1: # If success:
                                img1 = img1_
                                temp_success.append(temporarily_not_stitched[0])
                                temporarily_not_stitched = []
                        
                        if filter_stitched_images_to_correct_for_stitching_process: 
                            # Filter to remove artefacts from stitching
                            
                            img1 = median_filter(img1, size = 2)
                
                        if len(temporarily_not_stitched) > 0:
                            
                            unsuccessfully_stitched.append(np.sort(np.array(temporarily_not_stitched)))
    
                        if len(temp_success) > 0:
    
                            successfully_stitched.append(np.sort(np.array(temp_success)))
    
                        if np.shape(img1) != (0,): 
                            
                            stitched_particle_images.append(img1)
    
    
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%% 2 PARTICLE IMAGES STITCHED TOGETHER: %%%%%%%%%%%%%%%%%%%%%%%
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    else: # If only 2 particle images
    
                        idx0, idx1 = particle_clusters[cl][0], particle_clusters[cl][1]  
    
                        img0 = self.get_depadded_particle_image(idx0)
                        img1 = self.get_depadded_particle_image(idx1) 
    
                        # For intensity rescaling:
                        minI = np.min([np.min(img0), np.min(img1)])
                        maxI = np.max([np.max(img0), np.max(img1)])
    
                        img1_, successful_stitching = self.Images.stitch_two_particle_images(
                            particle_label1 = idx0,
                            particle_label2 = idx1,
                            shift_threshold = shift_threshold,
                            remove_non_matched_region = remove_non_matched_region,
                            success_threshold = stitching_ncc_threshold)
                        
                        if successful_stitching: img1 = img1_
    
                        elif not successful_stitching and directly_stitch_edge_particles:
                            # First attempt failed, try concatenate the images:
                            
                            # The following will concatenate the particle images but with padding:
                            """
                            # Make the two images into the same size:
                            pim1, pim2 = _image_utils._copy_images_into_common_array_shape(img0, img1)
                            
                            _, (loc0, loc1), axis = _image_utils._check_two_stitched_particle_images_concatenation_compatibilities(
                                list_of_labels0 = [idx0],
                                list_of_labels1 = [idx1],
                                particle_map = self.Images.particle_map.data)
                        
                            
                            # Remove padding along a specific axis:
                            pim1 = _image_utils._remove_padding_along_axis(pim1, axis = axis)
                            pim2 = _image_utils._remove_padding_along_axis(pim2, axis = axis)
                            
                            # Concatenate:
                            img1 = _image_utils._concatenate_two_edge_images(pim1, pim2, loc0, loc1)
                            """

                            img1 = self.Images.concatenate_two_cropped_particle_images(
                                particle_label1 = idx0,
                                particle_label2 = idx1)
    
                            successful_stitching = True
        
                        if successful_stitching:
    
                            if filter_stitched_images_to_correct_for_stitching_process: 
                                
                                img1 = median_filter(img1, size = 2)
    
                            stitched_particle_images.append(
                                rescale_intensity(
                                    img1.copy(), 
                                    out_range = (minI, maxI)
                                ).astype(self.Images.particle_images.data.dtype))
    
                            successfully_stitched.append(np.sort(np.array([idx0, idx1])))
    
                        else: 
                            
                            unsuccessfully_stitched.append(np.sort(np.array([idx0, idx1])))
    
                # Save the final results as hidden attributes, as pairs of particle images will become correted in the following, 
                # and their corresponding information will get updated.
                # Information about stitched images with > 2 particle images will be stored in an open attribute.
    
                unlabelled_stitched_particles_indices = []
                relabelled_stitched_particles_indices = []
                unlabelled_particle_images = []
                relabelled_particle_images = []
                
                for cl in range(len(successfully_stitched)):
                    
                    if len(successfully_stitched[cl]) > 2: 
                        #print('1', successfully_stitched[cl])
                        unlabelled_stitched_particles_indices.append(successfully_stitched[cl])
                        unlabelled_particle_images.append(stitched_particle_images[cl].copy())
                    else: 
                        #print('2', successfully_stitched[cl])
                        relabelled_stitched_particles_indices.append(successfully_stitched[cl])
                        relabelled_particle_images.append(stitched_particle_images[cl].copy())
                
                argsort = np.argsort(np.asarray(relabelled_stitched_particles_indices), axis = 0)[:,0]
                relabelled_stitched_particles_indices = np.asarray(relabelled_stitched_particles_indices)[argsort]
                _ = []
                for arg in argsort: _.append(relabelled_particle_images[arg])
                relabelled_particle_images = _
    
                # Particle images:
                self.Images._successfully_stitched_particle_images = stitched_particle_images
    
                self.Images.unrelabelled_stitched_particle_images = unlabelled_particle_images
    
                self.Images.relabelled_stitched_particle_images = relabelled_particle_images
                
                # Particle labels:
                self.metadata.Particles._successfully_stitched_particles_labels = successfully_stitched
    
                # Update the particle clusters, i.e. those remaining
                self.metadata.Particles.cropped_particle_clusters = unlabelled_stitched_particles_indices
    
                self.metadata.Particles.relabelled_stitched_particle_labels = relabelled_stitched_particles_indices
    
                self.metadata.Particles.unsuccessfully_stitched_particle_labels = unsuccessfully_stitched
    
                # Print the success rate
                number_of_cropped_particles = self.metadata.Particles.cropped_particles.shape[0]
                number_of_not_stitched_particles = 0
                for i in range(len(unsuccessfully_stitched)): 
                    number_of_not_stitched_particles += len(unsuccessfully_stitched[i])
                number_of_stithced_particles = number_of_cropped_particles - number_of_not_stitched_particles
                    
                print(f"\nSuccessful stitching: {number_of_stithced_particles}/{number_of_cropped_particles} ({np.format_float_positional(100 * number_of_stithced_particles/number_of_cropped_particles, 2)} %)")

                # Update particle information ONLY for particle pairs!
                if update_particle_properties:
                    
                    self._correct_successfully_stitched_particle_pairs()
    
            else: print('No cropped particles have been identified.')  

    def _cluster_group_of_cropped_particles(self,
                                            conditions : dict,
                                            update_particle_properties : bool = True):
        """Group particle images and labels together that are most likely corresponding to the same 
        particle, but have been separated due to particle cropping.

        

        Parameters
        ---------
        conditions
            dictionary of conditions to threshold particle images and segment them.
        update_particle_properties
            Whether to update the particles' labels and chemical composition.
            By default: True
        """
        
        if not hasattr(self.metadata, 'Particles'):
            
            raise AttributeError("The object has no Particles' class attribute. See the function *.load_images()")
        
        if not hasattr(self.metadata.Particles, 'cropped_particle_clusters'):
            
            raise AttributeError("The object has no, or has not identified cropped particles. See the function *.identify_cropped_particles().")

        # Save the info
        print('\033[1mSaved information:\033[0m')
        print('New particle images are saved in a list of dictionaries:')
        print('\t*.Images.grouped_cropped_particle_images[INDEX][KEY] \n\t~ *.metadata.Particles.cropped_particle_clusters[INDEX]')
        print('\t~ *.metadata.Particles.grouped_cropped_particle_clusters[INDEX][KEY]')
        self.Images.grouped_cropped_particle_images = []
        
        self.metadata.Particles.grouped_cropped_particle_clusters = []
        
        self.Images._grouped_cropped_particle_cluster_masks = []

        for cl in range(len(self.metadata.Particles.cropped_particle_clusters)):

            # Get particle labels grouped together
            clustered_particles = self.metadata.Particles.cropped_particle_clusters[cl].copy()

            # Get stitched image corr. to the group of particle labels
            stitched_particle_image = self.Images.unrelabelled_stitched_particle_images[cl].copy()

            # group particle images according to the stitched particle image:
            labels, masks, new_particle_images = self.Images.cluster_group_of_cropped_particles(
                list_of_cropped_particle_labels = clustered_particles,
                st_im = stitched_particle_image,
                conditions =  conditions
            )
            
            # Save the info
            self.metadata.Particles.grouped_cropped_particle_clusters.append(labels)

            self.Images._grouped_cropped_particle_cluster_masks.append(masks)

            self.Images.grouped_cropped_particle_images.append(new_particle_images)

        if update_particle_properties:
            
            self._correct_successfully_stitched_group_of_particles()

    def _correct_successfully_stitched_particle_pairs(self):
        """Following stitching of particles, only pair of particles will be relabelled, their chemistry will be updated,
        and their corresponding particle images will be updated. Stitched particles with more than two labels will 
        not be updated, as these are more challenging to automatically relabel. This should be done manually.
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("No images have been read yet. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("No information about the particles' spatial distribution has been stored. See the function *.identify_particle_regions()")
        
        if not hasattr(self.metadata.Particles, '_successfully_stitched_particles_labels'): 
            
            raise AttributeError("No cropped particle pairs have been identified. See function *.stitch_cropped_particles()") 

        if len(self.metadata.Particles.relabelled_stitched_particle_labels) > 0:
        
            from tqdm import tqdm
    
            # Particle pairs are easy to automatically correct for. > 2 particles on the other hand...
            pairs = self.metadata.Particles.relabelled_stitched_particle_labels
    
            # Identify the largest stitched particle image shape to assess whether the signal 'particle_images' need reshaping.
            max_pair_image_shape = [0,0] # Max stitched particle image shape
            #pairs_mask = np.zeros(len(self.Images._particle_images_stitched), bool)
            for pim in self.Images.relabelled_stitched_particle_images:
                temp_shape = pim.shape 
                if temp_shape[0] > max_pair_image_shape[0]: max_pair_image_shape[0] = temp_shape[0]
                if temp_shape[1] > max_pair_image_shape[1]: max_pair_image_shape[1] = temp_shape[1]
    
            # Reshape particle images if needed:
            self.Images._reshape_particle_images(tuple(max_pair_image_shape))
    
            num_particles = self.number_of_particles
    
            # Array masks defining the particles to keep and not after stitching particles
            keep_arr = np.zeros(num_particles, bool)
            del_arr = np.zeros(num_particles, bool)
    
            pimage_shape = self.Images.particle_image_shape

            # Appending stitched images before reshaping them to become
            # compatible with the particle images' signal
            _particle_images_stitched = []
    
            # Hide old cropped particles attributes
            if not hasattr(self.Images, '_initial_cropped_particles_map'):
            
                self.Images._initial_cropped_particles_map = self.Images.cropped_particles_map.deepcopy()
            
            
            for pair, im_idx in tqdm(zip(pairs, np.arange(len(pairs))), 
                                     desc = "Correcting pair of cropped particles' composition and labels",
                                     total = len(pairs)):
    
                # Store which particles to update and which particles to remove from the class:
                keep_arr[np.where(self.Images.unique_particle_labels == pair[0])] = True # total number of particles
                del_arr[np.where(self.Images.unique_particle_labels == pair[-1])] = True # total number of particles
    
                # Update the list of cropped particles and keep the remaining labels in larger clusters:
                self.metadata.Particles.cropped_particles = np.delete(
                    self.metadata.Particles.cropped_particles, self.metadata.Particles.cropped_particles == pair[0])
                self.metadata.Particles.cropped_particles = np.delete(
                    self.metadata.Particles.cropped_particles, self.metadata.Particles.cropped_particles == pair[-1])
                
                # Update particle labels in the particle_map 
                self.Images.particle_map.data[np.where(self.Images.particle_map == pair[-1])] = pair[0]
    
                # Remove the cropped particles from the cropped_particles_map
                self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == pair[0]] = 0
                self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == pair[-1]] = 0
    
                # Update chemistry by averaging:
                self.Particles.composition[:, 
                    np.where(self.Images.unique_particle_labels == pair[0])] = np.mean(
                    [self.Particles.composition[:, 
                        np.where(self.Images.unique_particle_labels == pair[0])], self.Particles.composition[:,
                        np.where(self.Images.unique_particle_labels == pair[1])]], axis = 0)

                # Relabel classification if the classes are unequal
                cl0 = self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[0])]
                cl1 = self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[-1])]
                
                if cl0 != cl1:
                    
                    self.Particles.classes[np.where(self.Images.unique_particle_labels == pair[0])] = cl0 + cl1
    
                # Setting negative composition to 
                #self.Particles.composition[:, pair[-1]-1][:] = -1 
    
                # Reshape the stitched particle images into a shape that is compatible with the class's particle images:
                _particle_images_stitched.append(_image_utils._put_array_content_into_larger_array(
                    self.Images.relabelled_stitched_particle_images[im_idx], pimage_shape))
    
            # Update the list of unique particle labels
            self._update_particle_label_list()
            
            # Setting negative geometries to update these manufally afterwards. 
            # Geometries no longer of interest are removed.
            for geom in self.Particles.particle_geometry.keys():
    
                self.Particles.particle_geometry[geom][keep_arr] = -1
    
                self.Particles.particle_geometry[geom] = self.Particles.particle_geometry[geom][~del_arr] 
    
            # Update the total number of particles:
            self.number_of_particles -= np.sum(del_arr)

            self.Particles.classes = self.Particles.classes[~del_arr]

            self.is_classified = self.is_classified[~del_arr]
    
            # Update the array of particles' chemical composition
            self.Particles.composition = self.Particles.composition[:, ~del_arr].reshape((len(self.get_identified_elements()), 
                                                                                          self.number_of_particles))
    
            self.update_particles_composition_shape()
    
            self._update_particles_concentration()
    
            self.Images.particle_images.data[keep_arr] = np.asarray(_particle_images_stitched)
    
            self.Images.particle_images = Signal2D(self.Images.particle_images.data[~del_arr])
    
            self.metadata.Particles.label_name = list(np.asarray(self.metadata.Particles.label_name)[~del_arr])

        else: print('No cropped particles have been stitced')
    
    def _correct_successfully_stitched_group_of_particles(self):
        """Following stitching of particles, only pair of particles will be relabelled, their chemistry will be updated,
        and their corresponding particle images will be updated. Stitched particles with more than two labels will 
        not be updated, as these are more challenging to automatically relabel. This should be done manually.
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("No images have been read yet. See the function *.load_images()")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("No information about the particles' spatial distribution has been stored. See the function *.identify_particle_regions()")
        
        if not hasattr(self.metadata.Particles, 'grouped_cropped_particle_clusters'): 
            
            raise AttributeError("No cropped particle pairs have been identified. See function *.stitch_cropped_particles()")

        # Iterate through all the grouped cropped particle clusters:
        if len(self.metadata.Particles.grouped_cropped_particle_clusters) > 0:
        
            from tqdm import tqdm
    
            # Particle pairs are easy to automatically correct for. > 2 particles 
            # on the other hand...
            group = self.metadata.Particles.grouped_cropped_particle_clusters
    
            # Identify the largest stitched particle image shape to assess whether the signal 
            # 'particle_images' need reshaping.
            max_pair_image_shape = [0,0] # Max stitched particle image shape
            for pim in self.Images.unrelabelled_stitched_particle_images:
                temp_shape = pim.shape 
                if temp_shape[0] > max_pair_image_shape[0]: max_pair_image_shape[0] = temp_shape[0]
                if temp_shape[1] > max_pair_image_shape[1]: max_pair_image_shape[1] = temp_shape[1]
    
            # Reshape particle images if needed:
            self.Images._reshape_particle_images(tuple(max_pair_image_shape))
    
            num_particles = self.number_of_particles
    
            # Array masks defining the particles to keep and not after stitching particles
            keep_arr = np.zeros(num_particles, bool)
            del_arr = np.zeros(num_particles, bool)

            changed_particles_labels = []
    
            pimage_shape = self.Images.particle_image_shape
    
            # Hide old cropped particles attributes
            if not hasattr(self.Images, '_initial_cropped_particles_map'):
                
                self.Images._initial_cropped_particles_map = self.Images.cropped_particles_map.deepcopy()

            for cluster, listPos in tqdm(zip(self.metadata.Particles.grouped_cropped_particle_clusters,
                                            np.arange(len(group))),
                                         desc = "Correcting group of cropped particles' composition and labels",
                                         total = len(group)):

                unique_particle_labels = np.unique(cluster)
                
                for cl in unique_particle_labels:

                    # Get the corr. particle labels
                    plabels = self.metadata.Particles.cropped_particle_clusters[listPos][cluster == cl]

                    # Save the particle labels that are kept 
                    changed_particles_labels.append(plabels[0])

                    temp_classes = []

                    keep_arr[np.where(self.Images.unique_particle_labels == plabels[0])] = True
                    
                    for plabel in plabels:
    
                        # Store which particles to update and which particles to remove from the class, 
                        # and update particle labels in the particle_map, by keeping the first label
                        if plabel != plabels[0]:
                            
                            del_arr[np.where(self.Images.unique_particle_labels == plabel)] = True 
                            
                            self.Images.particle_map.data[np.where(self.Images.particle_map == plabel)] = plabels[0]
            
                        # Remove the cropped particles from the cropped_particles_map
                        self.Images.cropped_particles_map.data[self.Images.cropped_particles_map == plabel] = 0

                        # Relabel classification if the classes are unequal
                        temp_classes.append(
                            self.Particles.classes[
                                np.where(self.Images.unique_particle_labels == plabel)
                                ]
                        )

                        # Delete the label from the cropped_particles' list:
                        self.metadata.Particles.cropped_particles = np.delete(
                            self.metadata.Particles.cropped_particles, 
                            self.metadata.Particles.cropped_particles == plabel)
                        
                    unique_classes = np.unique(np.asarray(temp_classes))

                    # Set a unique class name for the new particle
                    if len(unique_classes) > 1:

                        class_name = ''

                        for cl in unique_classes: class_name += cl

                    else: class_name = unique_classes[0]
                        
                    self.Particles.classes[
                        np.where(
                            self.Images.unique_particle_labels == plabels[0])] = class_name
                    
                    # Update chemistry by averaging:
                    self.Particles.composition[:, 
                        np.where(self.Images.unique_particle_labels == plabels[0])[0]] = np.mean(
                        [self.Particles.composition[:, 
                            np.where(self.Images.unique_particle_labels == plabel)[0]] 
                        for plabel in plabels], axis = 0)
                    
                    # Reshape the stitched particle images into a shape that is compatible with 
                    # the class's particle images:
                    self.Images.particle_images.data[
                        np.where(
                            self.Images.unique_particle_labels == plabels[0])[0]
                        ] = _image_utils._put_array_content_into_larger_array(
                            self.Images.grouped_cropped_particle_images[listPos][cl], 
                            pimage_shape)
                    
            # Delete the cropped particle clusters attribute if it's empty
            del self.metadata.Particles.cropped_particle_clusters

            self.metadata.Particles.updated_particles_from_group_of_cropped_particles = np.sort(changed_particles_labels)
            
            # Update the list of unique particle labels
            self._update_particle_label_list()
            
            # Setting negative geometries of the labels that are kept to update these manufally 
            # afterwards. Geometries no longer of interest are removed.
            for geom in self.Particles.particle_geometry.keys():
    
                self.Particles.particle_geometry[geom][keep_arr] = -1
    
                self.Particles.particle_geometry[geom] = self.Particles.particle_geometry[geom][~del_arr] 
    
            # Update the total number of particles:
            self.number_of_particles -= np.sum(del_arr)

            self.Particles.classes = self.Particles.classes[~del_arr]

            self.is_classified = self.is_classified[~del_arr]
    
            # Update the array of particles' chemical composition
            self.Particles.composition = self.Particles.composition[:, 
                ~del_arr].reshape((len(self.get_identified_elements()), 
                                   self.number_of_particles))
    
            self.update_particles_composition_shape()
    
            self._update_particles_concentration()
    
            #self.Images.particle_images.data[keep_arr] = np.asarray(_particle_images_stitched)
    
            self.Images.particle_images = Signal2D(self.Images.particle_images.data[~del_arr])
    
            self.metadata.Particles.label_name = list(np.asarray(self.metadata.Particles.label_name)[~del_arr])

        else: print('No group of cropped particles have been clustered.')
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%% OPEN GENERAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def get_navigation_scale(self):
        """Return the stored navigation scale if it has been set"""

        if not hasattr(self.metadata, 'navigation_scale'):

            raise AttributeError("The scale hasn't been set yet. See the function *.set_navigation_scale()")

        if self.metadata.navigation_scale == 1:

            warnings.warn("The navigation scale is currently equal to 1. Make sure this is the correct one.")
        
        return self.metadata.navigation_scale
    
    def set_acceleration_voltage(self, HT):
        """Set the acceleration voltage used during particle analysis
        to the object's metadata

        Parameters
        ----------
        HT 
            High tension/acceleration voltage as integer or float
        """

        if not type(HT) in (float, int): raise TypeError(f"Provided high tension {HT} is not valid. Make sure it is a real number")
        
        self.metadata.Acquisition_settings.acc_voltage = float(HT)

    def set_matrix_composition(self, matrix_composition, unit):
        """Set the matrix chemcial composition for reference. 

        Parameters
        ----------
        matrix_composition
            Chemical composition of the matrix to facilitate identification of false particles. 
            A dictionary with elements as key arguments.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.print_identified_elements()
        ['C','O','Al','Fe']
        
        >>> s.set_matric_composition(matrix_composition = [0.2, 0.5, 95, 4.3] # Equivalent to {'C' : 0.2, 'O' : 0.5, ... 'Fe' : 4.3}
                                    )
        >>> s.get_matrix_composition()
        [0.2, 0.5, 95, 4.3]
        """
        #print("FIX ME! MATRIX COMPOSITION SHOULD BE SEPARATE FROM PARTICLES' COMPOSITION")
        num_elements = len(self.get_identified_elements())
        
        if type(matrix_composition) == dict:
            
            arg_keys = list(matrix_composition.keys())

            for key in arg_keys:

                if key not in ELEMENTS: raise ValueError(f"Element {key} is not recognised.")

        else: 
            
            raise TypeError(f"Matrix composition argument type ({type(matrix_composition)}) is not a valid type. Provide a dictionary.")
        
        m_comp = [matrix_composition[elem] for elem in matrix_composition.keys()]

        m_sum = np.sum(m_comp)

        if m_sum < 0.0 or m_sum > 100.0001: 
            
            raise ValueError(f"The provided composition is not normalised (sum({m_comp}) = {m_sum})") 

        elif m_sum != 100.0: 

            warnings.warn(f'Normalising the provided matrix composition from a total of {m_sum} % to 100%.\n')

            m_comp = [100.0 * matrix_composition[elem] / m_sum for elem in matrix_composition.keys()]
        
        #m_comp = np.zeros(num_elements)
        for elem in matrix_composition.keys():
        
            if elem not in self.get_identified_elements(): warnings.warn(f"\nElement {elem} is not part of the particles' chemistry.")

            #m_comp[self.get_identified_elements().index(elem)] = matrix_composition[elem]
        
        if m_sum == 1.0: m_comp *= 100

        # Change unit if necessary
        current_unit = self.metadata.chemical_unit
        
        if current_unit in AVAILABLE_UNITS[0]: current_unit = 0 # at.%
        else: current_unit = 1 # wt.%
        if unit in AVAILABLE_UNITS[0]: matrix_unit = 0 # at.%
        else: matrix_unit = 1 # wt.%

        if current_unit != matrix_unit:

            if current_unit == 0 and matrix_unit == 1: m_comp = np.round(weight_to_atomic(m_comp, self.Particles.elements), decimals = 2)

            else: m_comp = np.round(atomic_to_weight(m_comp, self.Particles.elements), decimals = 2)
        
        self.metadata.Particles.matrix_composition = m_comp
        
        self.metadata.Particles.matrix_elements = list(matrix_composition.keys())

        print(f"Storing matrix composition in the class' metadata.Particles attribute matrix_composition.")

    def set_navigation_scale(self, scale, unit):
        """Set the image scale

        Parameters
        ----------
        scale
            Image scale in unit/px (float)
        unit
            string
        """

        self.metadata.navigation_scale = scale

        self.metadata.navigation_unit = unit
        
        if hasattr(self, 'Images'):

            nav_shape = self.Images.navigation_shape

            sig_shape = self.Images.signal_shape

            if len(nav_shape) > 1:

                self.Images.navigation_signal.axes_manager[1].scale = scale * sig_shape[0]
                
                self.Images.navigation_signal.axes_manager[0].scale = scale * sig_shape[1]

                self.Images.navigation_signal.axes_manager[0].units = unit

                self.Images.navigation_signal.axes_manager[1].units = unit

            else:

                self.Images.navigation_signal.axes_manager[0].scale = scale * nav_shape[0] * sig_shape[0]

                self.Images.particle_images.axes_manager[0].units = unit

            self.Images.navigation_signal.axes_manager[-1].scale = scale

            self.Images.navigation_signal.axes_manager[-1].units = unit

            self.Images.navigation_signal.axes_manager[-2].scale = scale

            self.Images.navigation_signal.axes_manager[-2].units = unit

            self.Images.particle_images.axes_manager[-1].scale = scale

            self.Images.particle_images.axes_manager[-1].units = unit

            self.Images.particle_images.axes_manager[-2].scale = scale

            self.Images.particle_images.axes_manager[-2].units = unit

            if hasattr(self.Images, 'particle_map'):

                self.Images.particle_map.axes_manager[-1].scale = scale

                self.Images.particle_map.axes_manager[-1].units = unit
    
                self.Images.particle_map.axes_manager[-2].scale = scale
    
                self.Images.particle_map.axes_manager[-2].units = unit
    
    def change_chemical_unit(self):
        """Change the chemical unit of the class"""
        
        current_unit = self.metadata.chemical_unit

        # Change from wt.% to at.%
        if current_unit in AVAILABLE_UNITS[1]:

            self.Particles.composition = np.round(weight_to_atomic(self.Particles.composition, self.Particles.elements), decimals = 2)

            if hasattr(self.metadata.Particles, 'matrix_composition'): 
                
                self.metadata.Particles.matrix_composition = np.round(weight_to_atomic(self.metadata.Particles.matrix_composition, self.metadata.Particles.matrix_elements), decimals = 2)

            self.Particles.chemical_unit = '[at.%]'

            self.metadata.chemical_unit = '[at.%]'

        elif current_unit in AVAILABLE_UNITS[0]:

            self.Particles.composition = np.round(atomic_to_weight(self.Particles.composition, self.Particles.elements), decimals = 2)

            if hasattr(self.metadata.Particles, 'matrix_composition'): 
                
                self.metadata.Particles.matrix_composition = np.round(atomic_to_weight(self.metadata.Particles.matrix_composition, self.metadata.Particles.matrix_elements), decimals = 2)

            self.Particles.chemical_unit = '[wt.%]'

            self.metadata.chemical_unit = '[wt.%]'
        
        else: print(f"{current_unit} is not recognised as a chemical unit.")

    # Classified particles
    def classified_particles(self):
        """Returns an array of classified particles"""
        return self.Particles.classes != 'Unclassified'

    def print_max_composition(self):
        """Print the maximum chemical composition of each element"""
        max_comp = np.expand_dims(np.max(self.Particles.composition, axis=1), axis=1)
        utils.print_particles_property(max_comp,  
                                       header = self.get_identified_elements(), 
                                       label = [f'Max [{self.metadata.chemical_unit}]'])
        
    def print_min_composition(self):
        """Print the maximum chemical composition of each element"""
        min_comp = np.expand_dims(np.min(self.Particles.composition, axis=1), axis=1)
        utils.print_particles_property(min_comp,
                                       header = self.get_identified_elements(), 
                                       label = [f'Min [{self.metadata.chemical_unit}]'])

    def print_unclassified_particles_composition(self):
        """Print a table of the unclassified particles' chemistry"""
        utils.print_particles_property(self.Particles.composition[:,~self.is_classified],
                                       label = self.metadata.Particles.label_name[~self.is_classified],
                                       header = self.get_identified_elements())

    def print_classified_particles_composition(self):
        """Print a table of the unclassified particles' chemistry"""
        utils.print_particles_property(self.Particles.composition[:,self.is_classified],
                                       label = self.Particles.classes[self.is_classified],
                                       header = self.get_identified_elements())

    def print_class_composition(self, class_name):
        """Print the chemical composition of the particles classified as class_name"""
        class_name = str(class_name)
        
        if class_name not in self.get_unique_particle_classes(): raise ValueError(f"Class name {class_name} is not recognised")

        class_instances = self.Particles.classes == class_name
        
        utils.print_particles_property(self.Particles.composition[:,self.is_classified],
                                       label = self.Particles.classes[self.is_classified],
                                       header = self.get_identified_elements())

    def print_matrix_composition(self):
        """Print the matrix' composition if it has been allocated the object.
        """
        if not hasattr(self.metadata.Particles, 'matrix_elements'): raise ValueError("The object's matrix elements is not set yet. See *.set_matrix_composition()")

        if not hasattr(self.metadata.Particles, 'matrix_composition'): raise ValueError("The object's matrix composition is not set yet. See *.set_matrix_composition()")

        if len(self.metadata.Particles.matrix_composition) == 0: warnings.warn('The matrix composition is empty')

        utils.print_particles_property(np.expand_dims(self.metadata.Particles.matrix_composition, axis = 1),
                                       header = self.metadata.Particles.matrix_elements,
                                       label = [self.metadata.chemical_unit])

    def get_removable_elements(self):
        """Based on the maximum composition stored in the object's Particles class attribute composition, 
        the function will return an array of elements to remove.
        """
        return np.asarray(self.get_identified_elements())[np.max(self.Particles.composition, axis = 1) == 0]
    
    def update_Image_shape(self):
        """Update the Image class' shape.
        """
        self.Images.shape = self.Images.navigation_shape + self.Images.signal_shape

    def get_particles_image_id(self, print_warning = True):
        """Get the image ids where the different particles are located
        
        Returns
        -------
        image_ids 
            numpy array keeping track of the image ids where specific particles were identified.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> # Image numbers (image index = image number - 1) where particle 0,1,2,3,... are located.
        >>> s.get_particle_image_id()
        array([1,1,3,3,3,6,7,7,7,...]) 
        """
        if print_warning: 
            print('Note that the image ID from this function == (index ID + 1) as in the image naming.')
            print('(See *.metadata.Particles.label_name')
        
        image_ids = []

        for i in range(self.number_of_particles): image_ids.append(int(self.metadata.Particles.label_name[i].split('-',2)[1]))

        return np.array(image_ids)

    def create_navigation_grid_mask(self, edge_width = 1):
        """Create a mask to overlay on stitched images. The mask will highlight the image edges"""
        
        return self.Images.create_stitched_image_grid_mask(edge_width = edge_width)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%% IMAGE ANALYSIS %%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def get_particle_area_density(self):
        """Get the particle area density per image.

        Returns
        -------
        particle_area_density
            Array containing the number of particles per imaged area
        """
        image_area = np.prod(
            self.Images.signal_shape) * np.square(
            self.Images.navigation_signal.axes_manager[-1].scale)

        number_of_particles_per_image = self.Images.get_number_of_particles_per_image()

        return number_of_particles_per_image / image_area

    def get_number_of_classes_per_image(self):
        """Return the number of classified particles per class per image.

        Returns
        -------
        classes
            Classes which the numbers in number_of_classes corresponds to
        number_of_classes
            Number of particles per class per image
        """

        regridify = False
        
        if self.Images.is_gridified:

            grid_shape = self.Images.navigation_shape
            
            self.degridify_SEM_images()

            regridify = True

        classes = self.get_unique_particle_classes()
        
        number_of_classes = np.zeros((len(classes),) + self.Images.navigation_shape)

        pim_arr = self.get_particles_image_id(print_warning=False) - 1

        for im_id in range(np.max(grid_shape)):

            for cl in np.arange(len(classes)):
                
                number_of_classes[cl][im_id] = np.sum(
                    (self.Particles.classes == classes[cl])[np.where(
                        pim_arr == im_id
                    )]
                )

        number_of_classes = np.expand_dims(number_of_classes, axis = 2)
        
        if regridify:

            self.gridify_SEM_images(grid_shape)

            number_of_classes = _image_utils._gridify_nD_array_to_ND(
                number_of_classes,
                to_shape = (len(classes),) + grid_shape
            )

        return classes, number_of_classes

    def plot_bar_of_classes(self, 
                            bar_colour = 'tab:olive',
                            x_label_rotation = -25,
                            return_fig = False):
        """Plot a bar plot of the total number of classes

        Parameters
        ----------
        bar_colour 
            Colour of the bars in the bar plot
        x_label_rotation
            rotation of the x-labels
        return_fig
            Whether to return the figure or not
        """
        import matplotlib.pyplot as plt
        
        class_names = self.get_unique_particle_classes()
        
        num_particles = dict() #np.zeros((len(class_names)), int)
        
        for cname in class_names:
            
            num_particles[cname] = np.sum(self.Particles.classes == cname)
        
        x = np.arange(len(num_particles)) - 0.5
    
        fig, ax = plt.subplots(figsize = (8,8))
        
        ax.bar(x = x,
               height = num_particles.values(), 
               width = 1,
               color = bar_colour, 
               edgecolor = 'k')
        
        ax.set_xticks(
            ticks = x, 
            labels = num_particles.keys(),
            rotation = x_label_rotation,
            fontsize=14,
            ha = 'left')    
        
        plt.show()
        
        if return_fig: return fig
            
    def plot(self, colours = None, background_colour = 'whitesmoke'):
        """Plot the phase maps stored in Images class
        
        Parameters
        ----------
        colours 
            list of dictionary of colours
        background_colour
            Colour of non-particles (background)
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The object has no Images class attribute yet. See load_images()")

        if not hasattr(self.Images, 'phase_map'): 
            
            raise AttributeError("The class has no maps of the particles' region. See *.identify_particle_regions()")

        if len(self.Images.phase_map) == 0: 
            
            print('Phase maps need to be defined first. See function *.update_phase_maps()')

        else:
            
            if not colours is None:

                if type(colours) == type(dict()):

                    class_args =  self._check_class_arguments(list(colours.keys()))
                
                    if not class_args[0]: 
                        
                        warnings.warn(f"The provided classes {class_args[1]} in colour argument is not recognised.")

            self.Images.plot(
                unique_classes = self.get_unique_particle_classes(),
                colours = colours,
                background_colour = background_colour
            )

    def get_entire_phase_map(self):
        """Return the entire phase map as a numpy ndarray

        Returns
        -------
        phase_map
            numpy.ndarray representing the phase map. The labels corresponds to the Images' class' phase_map keys.
        phase_labels
            Phase label IDs representing the phase_map keys.

        """
        if not hasattr(self, 'Images'): raise AttributeError("Object doesn't have the Image class attribute. See *.load_images()")

        if len(self.Images.phase_map) == 0: print("The phase maps have not been set yet. See *.update_phase_maps()")

        else:

            return self.Images.get_entire_phase_map(self.get_unique_particle_classes())
                                                  
    def load_images(self, image_path, set_image_dtype = None):
        """Read the individual particle images, SEM images and the stitched image and allocate it to the object's
        Images class. Note that the image path is expected to point to the content with folders named as "View000x, 
        View0002, ..."

        Parameters
        ----------
        image_path 
            path to where the images are stored (string)
        set_image_dtype
            Whether to set the images to a data type different from the default one set by pyplot.imread
            (float64). Note that the argument data type will be checked if it will change the image data 
            or not before setting the data type. If it will change the data type, the default data type 
            is used. 

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        
        >>> s.load_images(path, set_image_dtype = np.uint8)
        >>> s.Images.navigation_signal.data.dtype
        dtype('uint8')
        
        >>> # Plot SEM images:
        >>> s.Images.navigation_signal.plot()
        """

        try: label_names = self.metadata.Particles.label_name

        except AttributeError: raise AttributeError("Label names is not stored in the object's metadata")
        
        if not (self.number_of_particles == len(label_names)): 
            
            raise AttributeError(f"Number of particles {self.number_of_particles} is not equvalent to the number of particle label names ({len(label_names)})") 
        
        self.Images = _images.Images(path = image_path, label_names = label_names, dtype = set_image_dtype)
        
        Image_shape = self.Images.navigation_signal.data.shape

        if np.shape(Image_shape) == (3,): nav_shape, sig_shape = (Image_shape[0],), Image_shape[1:]

        elif np.shape(Image_shape) == (4,): nav_shape, sig_shape = Image_shape[:2], Image_shape[2:]

        else: raise ValueError(f"The Images' shape <({nav_shape} | {sig_shape})> are not expected.")
        
        # Store the image shape information from the particle analysis...
        self.metadata.Acquisition_settings.navigation_shape = nav_shape

        self.metadata.Acquisition_settings.signal_shape = sig_shape

        # and the current status:
        self.Images.navigation_shape = nav_shape

        self.Images.signal_shape = sig_shape

        self.update_Image_shape()
        
        print(f"Setting navigation and signal shape to ({nav_shape[0]}|{sig_shape[0]}, {sig_shape[1]})")

    def get_depadded_particle_image(self, particle_label_id):
        """As the particle images have padded (zero-valued) regions so that they have the same shape,
        the current function is intended to remove the padded edges and return only the particle image.

        Parameters
        ----------
        particle_id
            Particle label in the particle_map signal. I.e. particle_id us located at the index position == particle_id - 1

        Returns
        -------
        depadded_particle_image 
            particle image with no pad
        """

        if not hasattr(self, 'Images'): raise AttributeError("No images have been loaded. See the function *.load_images()")

        else: return self.Images.get_depadded_particle_image(particle_label_id)
        
    def get_stitched_SEM_image(self, navigation_shape = None, 
                               horisontal_direction = 'r2l', 
                               vertical_direction = 't2b',
                               scale = None):
        """Stitch the individual SEM images into one large overview image.

        Parameters
        ----------
        navigation_shape 
            Shape which the images will be stitched into. Default None, as it will try to read the object's navigation_shape
            Obs! It's the same convention as matplotlib's real space convention, i.e. 4 images in horisontal direction and 3 images 
            in vertical direciton will be correctly stitched if navigation_shape = (3,4)
        horisontal_direction
            Particle analysis acquired images from right to left, hence the default argument r2l. Alternative is to read from l2r
        vertical_direction
            Particle analysis acquired images from top to bottom, hence the default argument t2b. Alternative is to read from b2t

        Returns
        -------
        stitched_image_array
            numpy.ndarray with the stitched images
        """
        
        if not hasattr(self, 'Images'): 
            
            raise ValueError("Images class must be allocated your object, and the SEM images need to be read.")
        
        if self.Images.is_gridified and navigation_shape is None: 
            
            navigation_shape = self.Images.navigation_shape

            sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

        elif not self.Images.is_gridified and navigation_shape is None:

            raise ValueError(f"Images are not gridified. Provide a valid navigation_shape different from '{navigation_shape}'")

        else:

            if self.Images.is_gridified: sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

            else: sem_images = self.Images.navigation_signal.data
            
        if np.prod(navigation_shape) != np.prod(self.metadata.Acquisition_settings.navigation_shape):

            raise ValueError(f"The navigation shape {navigation_shape} is not compatible with the number of analysed images ({self.metadata.Acquisition_settings.analysed_views})")
        
        stitched_im = _image_utils._stitch_images(sem_images,
                                             navigation_shape + self.metadata.Acquisition_settings.signal_shape,
                                             horisontal_direction = horisontal_direction,
                                             vertical_direction = vertical_direction)

        if scale is not None:

            from skimage.transform import rescale
            
            stitched_im = rescale(stitched_im, scale, anti_aliasing=False)
        
        return stitched_im
    
    def gridify_SEM_images(self, navigation_shape, flip_axis = 1):
        """Reshape the object's 3D array of SEM images into a 4D array: 2 stage coordinates and 2 SEM image coordinates.

        Parameters
        ----------
        navigation_shape
            Shape of the stage navigation (list or tuple)
        flip_axis 
            The axis which to flip (as particle analysis images are acquired from right to left, top to bottom)

        Example
        >>> import particle_analysis as pa
        >>> s = pa.load("Results.csv")
        >>> s.read_images(Image_path)
        >>> s.Images.navigation_signal.data.shape
        (20, 768, 1024) 
        >>> s.gridify_SEM_images(navigation_shape = (5,4))
        >>> s.Images.navigation_signal.data.shape
        (5,4,768,1024) # (Y,X,y,x)
        """
        
        navigation_shape = tuple(navigation_shape)
        
        if not hasattr(self, 'Images'): 
            
            raise AttributeError("Images class must be allocated the object, and the SEM images need to be read.")

        if np.prod(navigation_shape) != np.prod(self.metadata.Acquisition_settings.navigation_shape):

            raise ValueError(f"The navigation shape {navigation_shape} is not compatible with the number of analysed images ({self.metadata.Acquisition_settings.analysed_views})")
        
        #if 1 not in navigation_shape:
            
        # Make the signal into the initially read 3D array
        
        if navigation_shape != self.Images.navigation_shape and len(self.Images.navigation_shape) > 1:
            
            self.degridify_SEM_images(flip_axis = flip_axis)
        
        self.Images.navigation_signal = Signal2D(
            _image_utils._gridify_3D_array_to_4D(
                self.Images.navigation_signal.data,
                to_shape = navigation_shape + self.metadata.Acquisition_settings.signal_shape,
                flip_axis = flip_axis)
        )

        # Update the image navigation shape
        self.Images.navigation_shape = navigation_shape

        self.update_Image_shape()

        # Update the other 4D signals
        self.Images._gridify_2Dsignals()

        self.Images.is_gridified = True

        self._update_set_navigation_scale()

        #else: print(f"Provided grid shape {navigation_shape} is not valid.")

    def degridify_SEM_images(self, flip_axis = 1):
        """Make the 4D image signal into a 3D image signal
        This is the opposite function of gridify_SEM_images
        """

        self.Images.navigation_shape = self.metadata.Acquisition_settings.navigation_shape

        self.update_Image_shape()

        if self.Images.navigation_signal.data.shape != self.Images.shape:
    
            print(f"Degridifying array of shape {self.Images.navigation_signal.data.shape} -> {self.Images.shape}")
            
            self.Images.navigation_signal = Signal2D(
                _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data,
                                                flip_axis = flip_axis)
            )
    
            self.Images.is_gridified = False

            self.Images._degrifify_2Dsignals()

            self._update_set_navigation_scale()

        else: print("Signal is already degridified.")

    def set_particle_regions(self,
                             labelled_particle_map,
                             horisontal_labelling_direction = 'l2r',
                             vertical_labelling_direction = 't2b'):
        """As an altenrative to the much slower function *.identify_particle_regions(), apply the set conditions 
        during particle analysis to re-identify the particles in the SEM images. 

        Parameters
        ----------
        conditions
            Set of conditions (dict)
        label_particles 
            Whether to label the particles and not just make a binary mask of the particles
        labelling_direction
            Direction in which the particles are labelled in the particle analysis software. Default: right to left (r2l)??
        
        """
        if hasattr(self.Images, 'particle_map'): print("The particle regions have already been defined. Run \n\t>>> del [self].Images.particle_map\nto enable redefinition of the particle regions.")

        elif labelled_particle_map.shape != self.Images.navigation_shape + self.Images.signal_shape:

            print(f'The labelled particle map shape ({labelled_particle_map.shape}) is not compatible with the signal shape {self.Images.navigation_shape + self.Images.signal_shape}')

        elif np.unique(labelled_particle_map) != np.array([0,1]) or np.unique(labelled_particle_map)[-1] != self.number_of_particles:

            print(f"The number of unique particle IDs ({np.unique(labelled_particle_map[-1])}) is not compatible with the number of particles from the particle analysis ({self.number_of_particles})\n Providing a particle map with different number of particles than the number from the data acquisition can complicate the analysis.")
                  
        else:

            print('NOT WRITTEN YET')
            print('FIX labelling direction!!!')
            print('FIX PARTICLE IMAGE ACQUISITION')
            # THE PLAN HERE IS TO MAKE THE USED PROVIDE THE USED LABELLING DIRECTION, AND THE FUNCTION WILL CORRECT FOR THIS BY RELABELLING IN ACCORDANCE WITH Jeol's LABELLING SCHEME

            self.Images.particle_map = labelled_particle_map
                                
        
    def identify_particle_regions(self, 
                                  label_particles : bool = True,
                                  return_overlapping_pixels_map : bool = False,
                                  matrix_label : int = 0):
        """Identify the particle regions in each SEM image by running template matching.
        (See skimage.feature's function match_template). The particle map is stored in the
        Images' object attribute *.particle_map

        Parameters
        ----------
        label_particles 
            Whether to get a particle map as a boolean mask or a labelled map. 
            Note that the labelling is according to the particle's labels
        
        return_overlapping_pixels_map
            Whether to return a map containing the pixels where the particle images 
            are overlapping.
            
        Note that overlapping regions are shared between the particles (~equally) 

        Returns
        -------
        omap
            Map of identified particles' overlapping pixels, given True 
            return_overlapping_pixels_map argument. 
            

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(path)
        
        >>> overlapping_pixels = s.identify_particle_regions(return_overlapping_pixels_map = True)
        >>> s.Images.particle_map
        <Signal2D, title: , dimensions: (4,5|1024, 768)>
        """

        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The images from particle analysis must be loaded to map the particles. See the function *.load_images().")

        if not hasattr(self.Images, 'particle_images'):

            raise AttributeError("The object doesn't have images of particles. These are needed to map the particles' position in the SEM images.")

        if not hasattr(self.Images, 'navigation_signal'):

            raise AttributeError("The object doesn't have the acquired SEM images. These are needed to map the particles' position.")
        
        if len(self.Images.navigation_shape) == 2: sem_images = _image_utils._gridify_4D_array_to_3D(self.Images.navigation_signal.data)

        else: sem_images = self.Images.navigation_signal.data
        
        pmaps = list(_image_utils._get_map_of_particle_regions(
            particle_images = self.Images.get_group_of_depadded_particle_images(self.Images.unique_particle_labels),
            SEM_images = sem_images,
            SEM_image_IDs = self.get_particles_image_id(print_warning = False),
            label_particles = label_particles))
        
        if self.Images.is_gridified: 
            
            pmaps[0] = _image_utils._gridify_3D_array_to_4D(pmaps[0], to_shape = self.Images.shape)

            if return_overlapping_pixels_map: pmaps[1] = _image_utils._gridify_3D_array_to_4D(pmaps[1], to_shape = self.Images.shape)
        
        self.Images.particle_map = Signal2D(pmaps[0])

        self._update_particle_label_list(matrix_label)

        plabels = self.Images.unique_particle_labels.copy()

        # Check if some lables have been removed
        if self.number_of_particles != len(plabels):

            missing_labels = _utils._identify_missing_labels(
                current_labels = plabels, 
                desired_labels = np.arange(1, self.number_of_particles + 1)
            )
            
            if len(missing_labels) > 0:

                # Correct for the missing labels by insert single pixels:
                print("IMPROVE REINSERION OF PARTICLE REGIONS - CONSIDER 'DILATION'")
                correct = input('Correct for missing particle labels? ([y]/n)')

                if correct == '' or correct.upper() == 'Y':

                    print('Correcting for missing labels')

                    regridify = False

                    desired_label_list = np.arange(1, self.number_of_particles + 1)
                    
                    if self.Images.is_gridified:

                        regridify = True

                        grid_shape = self.Images.navigation_shape

                        self.degridify_SEM_images()

                    im_indices = self.get_particles_image_id(print_warning=False) - 1

                    for missing_label in missing_labels:

                        nav_index = im_indices[np.where(desired_label_list == missing_label)]

                        _pmap = _image_utils._get_single_particle_map(
                            SEM_image = self.Images.navigation_signal.inav[nav_index].data,
                            particle_image = self.Images.particle_images.inav[missing_label - 1].data
                        )

                        p_centre = np.round(
                            np.median(
                                np.where(_pmap), axis = 1
                            )
                        ).astype(int)

                        # Setting the centre pixel of the missing particle region as the particle label 
                        self.Images.particle_map.inav[nav_index].data[p_centre[0], p_centre[1]] = missing_label

                    if regridify:
                        
                        self.gridify_SEM_images(grid_shape)
                    
                    self._update_particle_label_list(matrix_label)

                    missing_labels = _utils._identify_missing_labels(
                        current_labels = self.Images.unique_particle_labels, 
                        desired_labels = np.arange(1, self.number_of_particles + 1)
                    )

        if return_overlapping_pixels_map: return pmaps[1]

    def identify_cropped_particles(self, edge_width = 1):
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

        Example
        -------
        >>> s = pa.load(filename)
        >>> s.load_images(image_path)
        >>> s.Images
        Particle analysis imags:
            SEM images: <(16 | 1536,2048) | Particle images: (1000 | (500,400))>

        >>> s.gridify_SEM_images(navigation_shape = (4,4))
        >>> s.Images
        Particle analysis imags:
            SEM images: <(4,4 | 1536,2048) | Particle images: (1000 | (500,400))>

        # Identify the particle regions
        >>> s.identify_particle_regions()
        >>> s.identify_cropped_particles(edge_width = 2)
        >>> s.Images.cropped_particles_map
        <Signal2D, title: , dimensions: (4,4|2048,1536)>
        """
        if not hasattr(self, 'Images'): raise AttributeError("The class doesn't have the attribute 'Images'. See the *.load_images() function.")      
            
        #cropped_particles_map = _image_utils._get_edge_particles_gridMap_from_labelled_particles_gridMap(
        #    self.Images.particle_map.data,
        #    edge_width = edge_width
        #)
        
        #self.Images.identify_cropped_particles(edge_width = edge_width)

        #print("Cropped particle map -> *.Images.cropped_particles_map.")
        #self.Images.cropped_particles_map = Signal2D(cropped_particles_map.copy())

        # Identify the cropped particles
        self.Images.map_cropped_particles(edge_width = edge_width)

        # Get the cropped particles' labels
        unique_particle_labels = np.unique(self.Images.cropped_particles_map)

        print("Cropped particle labels -> *.metadata.Particles.")
        self.metadata.Particles.cropped_particles = np.delete(unique_particle_labels, 
                                                              np.where(unique_particle_labels == 0)).astype(np.uint32)

        # Create a navigation grid mask:
        grid_mask = self.create_navigation_grid_mask(edge_width = edge_width)

        # Identify cropped particles at the grid edges:
        stitched_cropped_particles_map = utils.stitch_images(self.Images.cropped_particles_map.data, 
                                                             self.Images.navigation_shape)

        # Cluster the cropped particles
        edge_clusters = _image_utils._identify_particle_edge_clusters(stitched_cropped_particles_map,
                                                                      image_edges_grid = ~grid_mask)

        # Throw a warning if the only class is 'Unclassified'
        if len(self.get_unique_particle_classes()) == 1:

            if self.get_unique_particle_classes()[0] == 'Unclassified': 
                
                warnings.warn(f"\nThere are no classified particles. The cropped particles will not be corrected according to their classes.")
            
            else:

                print('The following has not been properly tested yet.')

                proceed = input('Proceed? ([y]/n')
                
                if proceed == '' or proceed.upper() == 'Y':
                    
                    edge_clusters = _utils._check_cropped_particle_class_compatability(classes = self.Particles.classes,
                                                                                       clusters = edge_clusters)

        print("Storing the clustered particles' corr. labels in *.metadata.Particles.cropped_particle_clusters")
        self.metadata.Particles.cropped_particle_clusters = edge_clusters

        # Hide a copy
        self.metadata.Particles._cropped_particle_clusters = edge_clusters

    def stitch_cropped_particles(self, 
                                 conditions : dict = {},
                                 stitching_ncc_threshold : float = 0.25,
                                 extra_pixels : float = 0.2,
                                 shift_threshold : float = 1.0,
                                 remove_non_matched_region : bool = False,
                                 directly_stitch_edge_particles : bool= True,
                                 filter_stitched_images_to_correct_for_stitching_process : bool = False,
                                 looping_progressbar : bool = False):
        """The function attempts to first stitch identified cropped particles, before correcting for 
        the particles composition and labels given that conditions is not empty.

        To correct the cropped particles' geometric properties, see the function *.update_particles_geometric_propertes()

        Parameters
        ----------
        stitching_ncc_threshold
            Float value describing the lowest allowable ncc score between two images' overlapping 
            region before stitching is considered successful.
            Default: 0.5.
        extra_pixels
            Float (percentage) defining how many extra pixels wrt. SEM image widht or height to 
            add to improve the chance of correct stitching. 
        shift_threshold 
            A number that assess whether the image stitching was successful or not by how much it was shifted 
            (Eucledian distance). Default: 1.0
        remove_non_matched_region 
            Whether to remove the region that was initially not part of the stored particle image from the 
            stitched image. By default: True
        directly_stitch_edge_particles 
            If particle stitching fails for some reason, concatenate the particle images directly. 
        filter_stitched_images_to_correct_for_stitching_process
            Whether to smooth the stitched images or not. A good reason to do this is to remove artefacts from the image shifting.
            See skimage.registration's phase_cross_correlation function.
        update_particle_properties
            Wheter to correct the cropped particle pairs' chemical composition and labels.
        """

        update_particle_properties = False

        if len(conditions) > 0: update_particle_properties = True

        # Stitch cropped particles and update pair of particles' labels and composition
        self._stitch_cropped_particles(
            stitching_ncc_threshold = stitching_ncc_threshold,
            extra_pixels = extra_pixels,
            shift_threshold = shift_threshold,
            remove_non_matched_region = remove_non_matched_region,
            directly_stitch_edge_particles = directly_stitch_edge_particles,
            filter_stitched_images_to_correct_for_stitching_process = filter_stitched_images_to_correct_for_stitching_process,
            looping_progressbar = looping_progressbar,
            update_particle_properties = update_particle_properties
        )

        # As some identified cropped particles might be grouped together, the following 
        # function attempts to segment these and group together labels/particles of the same origin
        self._cluster_group_of_cropped_particles(
            conditions = conditions,
            update_particle_properties = update_particle_properties
        )

    def update_particles_geometric_properties(self,
                                              conditions : dict,
                                              remeasure_properties = [
                                                  'Area [um²]',
                                                  'Maximum length [um]',
                                                  'Roundness',
                                                  'Orientation [degree]',
                                                  'Perimeter'], 
                                              labels : list | np.ndarray = [],
                                              return_single_particle_only : bool = True):
        """Update particles' geometric properties as provided by the remeasure_properties list. If the list is empty, the
        function will by default iterate through all pairs of stitched particles.
        
        The allowed properties are:
            -Area
            -Maximum length (maximum feret)
            -Roundness
            -Orientation
            -Perimiter

        See also https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        for more properties.

        The allowed string (?) conditions are 
            < 
            <=
            >
            >=
            min_size
            max_size
            binary_erosion
            binary_dilation
            fill_holes

        Parameters
        ---------
        conditions 
            Dictionary of conditions
        remeasure_properties
            list of geometric properties to remeasure
        labels
            Which particle (labels) to remeasure geometric properties. If the list is empty,
            the stitched pair of particles will be remeasured. 
        return_single_particle_only
            Whether to only return single particles from the particle image based on the 
            largest number of segmented pixels.
        Example
        -------
        s.stitch_cropped_particles()
        s.update_stitched_particles_geometric_properties(
            conditions = {'<' : 80, # Intensity lower than 80 is of interest
                          '>=' : 160, # Intensity lower than or equal to are of interest
                          '' : ,
        })
        """

        # For regionprops_table 
        property_translator = {
            'area [um²]' : 'area',
            'area' : 'area',

            'maximum length [um]' : 'feret_diameter_max',
            'maximum length' : 'feret_diameter_max',
            'max length' : 'feret_diameter_max',

            'roundness' : 'area_convex',

            'orientation [degree]' : 'orientation',
            'orientation' : 'orientation',

            'perimeter [um]' : 'perimeter',
            'perimeter' : 'perimeter',
        }

        class_translator = {
            'area' : 'Area [um²]',
            'area [um²]' : 'Area [um²]',

            'feret_diameter_max' : 'Maximum length [um]',
            'maximum length [um]' : 'Maximum length [um]',

            'area_convex' : 'Roundness',
            'roundness' : 'Roundness',

            'orientation' : 'Orientation [degree]',
            'orientation [degree]' : 'Orientation [degree]',

            'perimeter' : 'Perimeter [um]',
            'perimeter [um]' : 'Perimeter [um]',
        }

        if not hasattr(self, 'Images'): 
            
            raise AttributeError(f"The class doesn't have any particle images. See *.load_images().")

        if len(labels) == 0:

            if not hasattr(self.metadata.Particles, '_cropped_particle_clusters'): 
            
                raise AttributeError(f"The class' Particles class doesn't have any information about cropped particle clusters. See *.identify_cropped_particles().")
            
            if not hasattr(self.Images, '_successfully_stitched_particle_images'):
                
                raise AttributeError(f"The class doesn't have any stitched particles yet. See *.stitch_cropped_particles()")
            
            print(f'Correcting the geometric properties {remeasure_properties} for pair of stitched particles.')
            
            particle_labels = self.metadata.Particles.relabelled_stitched_particle_labels[:,0]

            # Add the new grouped particle labels if they exists.
            if hasattr(self.metadata.Particles, 'updated_particles_from_group_of_cropped_particles'):

                particle_labels = np.sort(
                    np.append(
                        particle_labels, self.metadata.Particles.updated_particles_from_group_of_cropped_particles
                    )
                )
        
        else: 
            
            print('Correcting geometric properties for the specified particles.')
            
            particle_labels = labels

        # Import libraries
        from tqdm import tqdm 
        from skimage.measure import label, regionprops_table

        if self.metadata.navigation_scale == 1:

            warnings.warn(f"The real space calibration scale is equal to one.")

        if 'roundness' in remeasure_properties:
            
            if len((set(remeasure_properties) & set(['perimeter [um]', 
                                                     'perimeter', 
                                                     'Perimeter [um]', 
                                                     'Perimeter',]))) == 0:
                
                # To estimate a particle's roundness, we need both the area_convex from the 
                # property_translator and the perimeter.
                remeasure_properties.append('perimiter')
        
        from skimage.measure import regionprops_table

        # Translate property arguments:
        properties = []
        for prop in remeasure_properties: properties.append(property_translator[prop.lower()])

        if not hasattr(self.Images, 'relabeled_stitched_particle_masks'):
            
            self.Images.relabelled_stitched_particle_masks = dict()
        
        for l in tqdm(particle_labels):

            if l not in self.Images.unique_particle_labels: 
                
                warnings.warn(f"Particle label {l} is not found among the unique particle labels.")

            else: 

                pmask = self.Images._create_particle_mask(
                    pimage = self.get_depadded_particle_image(l),              
                    conditions = conditions,                                      
                    return_single_particle_only = return_single_particle_only
                )

                lab_image = label(pmask)

                self.Images.relabelled_stitched_particle_masks[l] = lab_image
                
                if len(np.unique(lab_image)) > 2:
                    
                    warnings.warn(f"\n\nThe particle labelled as {l} has {len(np.unique(lab_image))} labels.\nThe particle's geometric properties is recommended to be manually corrected.")

                else:
                
                    props = regionprops_table(label_image = lab_image, 
                                              intensity_image=self.get_depadded_particle_image(l), 
                                              properties=properties,
                                              spacing = self.metadata.navigation_scale)
    
                    for key in remeasure_properties:
    
                        if 'roundness' in key.lower():
                            
                            props['Roundness'] = props['area_convex'] / props['perimeter']
                        
                        elif 'orientation' in key.lower():
    
                            props['Orientation [degree]'] = np.rad2deg(props['orientation'])
                        
                        
                        self.Particles.particle_geometry[
                            class_translator[key.lower()]
                            ][np.where(self.Images.unique_particle_labels == l)] = props[
                                property_translator[key.lower()]]

    def update_phase_maps(self, background_val = 0):
        """Create phase maps according to the classes. The individual phase maps
        will be stored as a dictionary in the "Images" class.

        Note that the shape of the phase maps are stored as a 3D array. Reshaping is 
        necessary to make it compatible with a gridified signal.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.load_images(image_path) # Necessary to identify the probed particles
        >>> s.gridify_SEM_images((5,4)) # Necessary to correctly shape the phase maps???
        >>> s.identify_particle_regions() # Get labelled particles
        >>> s.update_phase_maps() # Updating phase maps
        >>> s.plot() # Plotting phase maps with defualt or auto-generated rgb values
        """
        
        if not hasattr(self, 'Images'): 
            
            raise AttributeError("The class doesn't have Images attribute yet. See load_images function.")

        if not hasattr(self.Images, 'particle_map'): 
            
            raise AttributeError("The Images class attribute doesn't keep track of the particles' position. See *.identify_particle_regions()")

        # As we are iterating through self.get_particlse_image_id, we need to correct for the cropped particles
        # when making the phase maps.
        if hasattr(self.Images, 'cropped_particles_map'):

            correct_cropped_particles = True

        else: correct_cropped_particles = False

        num_classified_particles = len(self.Particles.classes) 
        
        num_unique_labels = len(self.Images.unique_particle_labels)

        if num_classified_particles != num_unique_labels:
            
            raise _errors.ShapeError(f"The number of classified particles ({num_classified_particles}) are different from the number of unique particle labels ({num_unique_labels}).")
        
        from tqdm import tqdm
        
        # Delete old phase maps
        if len(self.Images.phase_map.keys()) > 0: 
            
            del self.Images.phase_map

            self.Images.phase_map = dict()

        particle_map = self.Images.particle_map.data.copy()

        # Enable image iteration:
        if self.Images.is_gridified: 
            
            particle_map = _image_utils._gridify_4D_array_to_3D(particle_map)
        
        image_idxs = self.get_particles_image_id(print_warning = False) - 1

        classes = self.Particles.classes

        unique_classes = self.get_unique_particle_classes()
        
        for cl, counter in zip(unique_classes, np.arange(1, len(unique_classes) + 1)):

            print(counter,'/', len(unique_classes))

            # Temporary phase map
            phase_map = np.zeros_like(particle_map, bool)

            #class_indices = np.where(classes == cl)[0]
        
            #for img_idx in tqdm(image_idxs, desc=cl):
                
            #    unique_p_indices = np.unique(particle_map[img_idx])
                
                # Remove background
            #    unique_p_indices = np.delete(unique_p_indices, np.where(unique_p_indices == background_val))
                            
            #    for p_idx in unique_p_indices: 

            #        if (p_idx - 1) in class_indices:
                    
            #            phase_map[img_idx][np.where(particle_map[img_idx] == p_idx)] = True

            class_indices = np.where(classes == cl)
        
            for p_idx, img_idx in tqdm(zip(class_indices[0], 
                                           image_idxs[class_indices]), 
                                       desc=cl, 
                                       total = len(class_indices[0])):
                
                #phase_map[img_idx][np.where(particle_map[img_idx] == (p_idx + 1))] = True
                phase_map[img_idx][np.where(particle_map[img_idx] == self.Images.unique_particle_labels[p_idx])] = True
            
            self.Images.phase_map[cl] = phase_map.copy()

        # Correct for cropped particles:
        if correct_cropped_particles:

            print('NOT TESTED: Correcting for cropped particles')

            nav_shape = self.Images.navigation_shape

            self.degridify_SEM_images()

            relabelled_cropped_particles = self.metadata.Particles.relabelled_stitched_particle_labels[:,0]

            for lab in relabelled_cropped_particles:

                class_ = self.Particles.classes[np.where(self.Images.unique_particle_labels == lab)]

                self.Images.phase_map[class_[0]][np.where(self.Images.particle_map.data == lab)] = True

            self.gridify_SEM_images(navigation_shape = nav_shape)


        self.Images.update_phase_map_shape()

    def get_coloured_phase_map_on_SEM_images(self,
                                             colours,
                                             adjust_intensity = False,
                                             background_val = 0,
                                             iterate_through_phase_maps = True):
        """Tint gray-scale particle map with colours defined by the dictionary colours.

        Parameters
        ----------
        colours
            Colours dictionary
        background_val 
            Background value in labelled particle map (By default: 0)
        iterate_through_phase_maps
            Whether to mask array regions according to the phase_map arrays. 
            If the phase maps are large in size, it can be an idea to set it 
            to False to enable iteration through the SEM images that were found
            with particles.
            
        Returns 
        -------
        tinted particle rgb image
            a tinted gray-scale image according to the colours in colours argument

        Example
        -------

        >>> import particle_analysis as pa
        >>> s = pa.load('Results.csv')
        >>> s.load_images(image_path)
        
        >>> s.gridify_SEM_images(navigation_shape = (5,5))
        
        >>> s.identify_particle_regions(label_particles = True)
        Searching for the analysed particles in the SEM images...
        Overlapping particle regions will be shared.

        >>> phase_map = s.colour_particle_map(colours = ['red','blue','green'])
        >>> phase_map.shape 
        (5,5,768, 1024,3)
        """
        
        if type(colours) != type(dict()): raise TypeError("Provided colour argument is not a dictionary.")

        class_args =  self._check_class_arguments(list(colours.keys()))
        
        if not class_args[0]: 
            
            warnings.warn(f"The provided classes {class_args[1]} in colour argument is not recognised.")
        
        if not hasattr(self, 'Images'):
            
            raise AttributeError("The particle analysis images must be ead in order to get a map of the chemically mapped particles. See ")

        if not hasattr(self.Images, 'particle_map'):

            raise AttributeError("The particle images must be defined first to enable colouring...")

        from matplotlib import colors 
        from skimage.color import gray2rgb
        from tqdm import tqdm
        import sys

        gridified_signal = self.Images.is_gridified
        
        # Whether phase maps have been set
        updated_phase_maps = len(self.Images.phase_map)

        SEM_imgs = self.Images.navigation_signal.data.copy()

        if gridified_signal: SEM_imgs = _image_utils._gridify_4D_array_to_3D(SEM_imgs)
        
        # First option: make the coloured phase map according to phase_map arrays
        if gridified_signal and updated_phase_maps > 0 and updated_phase_maps == len(self.get_unique_particle_classes()) and iterate_through_phase_maps:

            from copy import deepcopy
            
            nav_shape = self.Images.navigation_shape        
    
            phase_maps = deepcopy(self.Images.phase_map)
    
            SEM_imgs = utils.stitch_images(SEM_imgs, nav_shape)
    
            for key in phase_maps.keys(): phase_maps[key] = utils.stitch_images(phase_maps[key], nav_shape)
    
            print('Getting SEM images as normalised rgb greyscale array ...')
            SEM_imgs = utils.greyscale_to_rgb(SEM_imgs)
    
            for class_ in tqdm(self.Images.phase_map.keys()): SEM_imgs[phase_maps[class_]] *= colors.to_rgb(colours[class_])

        else: # Second option: iterate through the images

            particle_map = self.Images.particle_map.data
            
            print('Getting SEM images as normalised rgb greyscale array ...')
            SEM_imgs = utils.greyscale_to_rgb(SEM_imgs)
            
            if gridified_signal: particle_map = _image_utils._gridify_4D_array_to_3D(particle_map)
            
            rgb_vals = np.zeros((self.number_of_particles, 3), dtype = np.float32)
            
            for cl in colours.keys(): rgb_vals[np.where(self.Particles.classes == cl)] = colors.to_rgb(colours[cl])
            
            image_idxs = self.get_particles_image_id(print_warning = False) - 1
    
            classes = self.Particles.classes
    
            unique_classes = self.get_unique_particle_classes()
            
            for cl, counter in zip(unique_classes, np.arange(1, len(unique_classes) + 1)):
    
                print(counter,'/', len(unique_classes))
    
                class_indices = np.where(classes == cl)
            
                for p_idx, img_idx in tqdm(zip(class_indices[0], image_idxs[class_indices]), desc=cl, total = len(class_indices[0])):
                    
                    SEM_imgs[img_idx][np.where(particle_map[img_idx] == (p_idx + 1))] *= rgb_vals[p_idx]

            print("\n\033[1mImage stitching can be done using the utils-function 'stitch_rgb_phase_map(phase_map, nav_shape = (X,Y).\033[0m")

        if adjust_intensity: 
            
            from skimage.exposure import rescale_intensity
            
            SEM_imgs = rescale_intensity(np.sqrt(SEM_imgs), out_range = (0,1))

        return SEM_imgs

    
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%% CHEMISTRY MANIPULATIONS %%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def threshold_particles_composition(self, lower_threshold):
        """Apply a lower threshold on the particles' chemical composition. The upper
        limit is 100 (%). I.e. element compositions < lower_threshold is set to zero.
        
        Parameters
        ----------
        lower_threshold 
            Lower threshold value (float) in percentage.
        """
        # Make sure the concentration is in the range [0, 100]:
        self._update_particles_concentration()
        
        self.Particles.composition[self.Particles.composition < lower_threshold] = 0.0

        self._update_particles_concentration()

    def update_particles_composition_shape(self):
        """Update the attribute element_shape"""
        self.particles_composition_shape = (len(self.Particles.elements), self.number_of_particles)
        
    def remove_element(self, element):
        """Remove a specific element, or list of elements from the object.
        The function will update the particles' element concentration.

        Parameters
        ----------
        element
            A single element (string) or a list of elements
        """
        if type(element) == str: element = [element]

        print(f"Removing elements {element}")

        available_elements = list(ELEMENTS.as_dictionary().keys())[1:]
        
        for elem in element:

            elem_index = self.Particles.elements.index(elem)

            if elem not in available_elements: 
                
                print(f"Element {elem} is not recognised and will be ignored.")

            else:

                if elem in self.get_identified_elements():
    
                    self.Particles.elements.remove(elem)
                    
                    self.Particles.composition = np.delete(self.Particles.composition, elem_index, 0)
    
                else:
    
                    print(f'Element {elem} has not been identified during the data acquisition.')
                    
            #if hasattr(self.metadata.Particles, 'matrix_composition'): 

            #    remove_from_matrix = input(f"Remove element {elem} from the matrix composition? (y/[n])")

            #    if remove_from_matrix.upper() == 'Y': 
    
            #       self.metadata.Particles.matrix_composition = np.delete(self.metadata.Particles.matrix_composition, elem_index)

        self._update_particles_concentration()

        self._update_particles_composition_shape()
    
    def get_identified_elements(self):
        """Get the stored elements in 'self.Particles' elements attribute as a list"""
        return list(self.Particles.elements)
        
    def print_identified_elements(self):
        """Print the stored elements in 'self.Particles'"""
        print(self.get_identified_elements())

        
    def get_particles_with_elements(self, list_of_elements):
        """Identify particles contaning ALL the elements specified in the list_of_elements argument. 
        Note that other elements might also be present in the particles.
        
        Parameters
        ----------
        list_of_elements
            List of elements or a string of a single element

        Returns
        -------
            Array of particles with elements as specified in the list_of_elements 
        """
        if type(list_of_elements) == str: list_of_elements = [list_of_elements]

        particles = np.ones((self.number_of_particles), bool)

        for elem in list_of_elements:

            particles *= (self.Particles.composition[self.Particles.elements.index(elem)] > 0)

        return particles

    def get_particles_containing_only_elements(self, list_of_elements, ignore_elements = []):
        """Identify particles contaning the elements specified in the list_of_elements argument. 
        Other elements might also be present in the particles.
        
        Parameters
        ----------
        list_of_elements
            List of elements or a string of a single element
        ignore_elements
            List of elements to ignore (f.ex. C and/or O) - NOT FIXED

        Returns
        -------
        particles 
            Array of particles only containing the specified elements
        """
        list_type = type(list_of_elements) 
        
        if list_type == str: list_of_elements = [list_of_elements]

        elif list_type not in (list, tuple): raise TypeError(f"{list_of_elements} is unexpected. Provide a list/tuple of elements (strings), or a single element (string)")

        if type(ignore_elements) == str: ignore_elements = [ignore_elements]
        
        is_empty = True # Whether to ignore certain elements or not.

        if len(ignore_elements) > 0: 

            # Check if the same element is in both lists.
            common_elements = set(list_of_elements) & set(ignore_elements)

            if len(common_elements) > 0: warnings.warn(f"The elements to ignore {common_elements} are also found in the list of elements to search for.") 
            
            is_empty = False

        num_elements = len(list_of_elements)

        # Make an array template reflecting the elements we're interested in
        template = [0] * len(self.Particles.elements)

        for elem in list_of_elements: template[self.Particles.elements.index(elem)] = 1

        # template shape will match the Particle's attribute composition 
        template = np.reshape(np.asarray(template * self.number_of_particles, bool), self.particles_composition_shape[::-1]).T
        
        # Array of particles where the elements are found (in addition to other elements)
        array_of_fits = np.sum((template * (self.Particles.composition > 0)), axis = 0) == num_elements 
        
        # An array of particles where the exact number of elements are found
        reference_array = np.sum(self.Particles.composition > 0, axis = 0) == num_elements

        return array_of_fits * reference_array

    def get_unique_particle_classes(self, return_list = True):
        """Print the object's sorted unique particle labels.
        
        Parameters
        ----------
        return_list
            Whether to return the list of unique labels. (Default: False)

        Returns
        -------
        labels
            List of unique particle labels if return_list is True.
        """
        labels = np.unique(list(set(self.Particles.classes.astype(list))))
        
        if return_list: return labels
        
        else: print(labels)
    
    def print_unique_particle_classes(self):
        """Print the object's sorted unique particle labels.
        
        Parameters
        ----------
        return_list
            Whether to return the list of unique labels. (Default: False)

        Returns
        -------
        labels
            List of unique particle labels if return_list is True.
        """
        self.get_unique_particle_classes(return_list = False)

    def identify_false_particles_by_chemistry(self):
        """Map the match score between particles and the provided matrix composition by running template 
        matching. A high match scores represents a high probability for being part of the matrix.

        Returns
        ------
        ncc_scores : np.ndarray
            Array with normalised cross-correlation scores representing the particles' chemistry match with the 
            set matrix composition.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(filename)
        >>> s.set_matrix_composition({'C' : 2.5, 'O' : 8.0, 'Al': 89.5}, unit = 'wt.%')
        >>> ncc_scores = s.identify_false_particles_by_chemistry()
        >>> ncc_scores
        array([0.97052249, 0.9584889 , 0.83429012, ..., 0.74499595, 0.5983186 ,
        0.92121971])

        >>> # Potential matches:
        >>> ncc_scores > 0.9999
        array([False, False, False, ..., False, False, False])
        """
        if not hasattr(self.metadata.Particles, 'matrix_composition'): raise ValueError("The class object doesn't contain information about the material's matrix composition")

        elif not hasattr(self.metadata.Particles, 'matrix_elements'): raise ValueError("The class object doesn't contain information about the material's matrix elements")

        else:

            #try: 
                
            #    from sklearn.metrics.pairwise import cosine_similarity

            #    print('Unfinished')
            #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

            #except ModuleNotFoundError: 

            print("Matching particles' chemical composition with the matrix composition using the NCC metric.")

            template = np.zeros((len(self.get_identified_elements())))
            
            matrix_elems = self.metadata.Particles.matrix_elements
            
            matrix_comp = self.metadata.Particles.matrix_composition
            
            identified_elems = self.get_identified_elements()
            
            for elem, conc in zip(matrix_elems, matrix_comp): template[identified_elems.index(elem)] = conc

            scores = _utils._template_match_1d(patterns = np.transpose(self.Particles.composition / 100),
                                                templates = template / 100,
                                                #num_particles = self.number_of_particles
                                               ).squeeze()

            return scores
            
            #false_positives = ncc_scores.squeeze() > (1 - (threshold / 100))

            #print(f"Number of identified false positives by chemistry: {np.sum(false_positives)}")

            #return false_positives

    def identify_false_particles_by_geometry(self,
                                             geometry = ['Feret diameter H. [um]',
                                                         'Feret diameter V. [um]']):
        """Identify potential false particles by clustering geometric properties defined by 
        geometry dictionary.

        Note that the data will be normalised according to the measurements largest value.

        Parameters
        ----------
        geometry
            dictionary with geometry key argument for the class' Particles attrubute particle_geometry

        Returns
        -------
        false_positives
            Array of false particles.

        Example
        -------
        >>> import particle_analysis as pa
        >>> s = pa.load(csv_filename)
        >>> s.identify_false_particles_by_geometry()
        
        """

        if not hasattr(self, 'Images'): raise ValueError("The class' attribute 'Images' is not yet defined. See *.load_images()")

        common_keys = set(geometry) & set(self.Particles.particle_geometry.keys())
        
        if set(geometry) == common_keys:

            from sklearn.cluster import DBSCAN

            # Normalisation constant
            max_ = max([np.max(self.Particles.particle_geometry[geom]) for geom in geometry])

            # Transposed normalised data
            X = np.asarray([self.Particles.particle_geometry[geom] / max_ for geom in geometry]).T

            clustering = DBSCAN(eps = 0.11, min_samples = 100).fit(X)
            
            if len(np.unique(clustering.labels_)) != 2: print('Multiple clusters were identified. Please adjust the clustering parameters.')
            
            else:
                arg_min = np.argmin([np.sum(clustering.labels_ == lab) for lab in np.unique(clustering.labels_)])

                label = np.unique(clustering.labels_)[arg_min]

                false_positives = clustering.labels_ == np.unique(clustering.labels_)[arg_min]
                
                print(f"Number of identified false positives by geometry: {np.sum(false_positives)}")

                return false_positives

        else: 
            
            uncommon_keys = []

            for key in geometry: 
                
                if key not in self.Particles.particle_geometry.keys(): uncommon_keys.append(key)
            
            raise TypeError(f"The keys {uncommon_keys} are not identified in the classæ Particles attribute particle_geometry- Valid arguments are: {list(self.Particles.particle_geometry.keys())}")

    #def identify_false_particles(self):
    #    """The function searches for false particles by chemistry and geometry"""
    #    print('\033[1m The function uses default values, and is therefore not recommended ...\033[0m')
    #    return self.identify_false_particles_by_chemistry() + self.identify_false_particles_by_geometry()

    def get_artificial_eds_map(self, 
                               bkgr_idx = 0,
                               Erange = 15.0, 
                               steps = 0.02,
                               stitch_signal = True):
        """Create a dummy EDS spectrum containing signals from the elements stored in element_list
        

        Parameters
        ----------
         bkgr_idx 
             Integer representing the background in the particle_map (0 by default)
         Erange
             float, Value of energy range. (20 keV by default)
         steps
             Energy resolution (0.02 keV by default)
         stitch_signal
             Whether to stitch the signal into a EDS map (True by default)
         Returns
         -------
         signal : exspy.signals.EDSSEMSpectrum
             Dummy EDS-SEM signal
         """

        if not hasattr(self.Images, 'particle_map'): raise AttributeError("The object doesn't have particle_map defined yet. See identify_particle_regions for instance.")
            
        #warnings.warn("The process is not optimised for large datasets...")
        
        from tqdm import tqdm

        im_ids = np.unique(self.get_particles_image_id(print_warning = False) - 1)
        
        regridify = False

        LINES = []

        if self.Images.is_gridified: 

            nav_shape = self.Images.navigation_shape

            regridify = True

            self.degridify_SEM_images()

        # Signal to store the artificial EDS map in
        signal = np.zeros((self.Images.navigation_shape) + self.Images.signal_shape + (int(Erange/steps),), dtype = np.uint8)

        for im in tqdm(im_ids):

            #labels = np.unique(self.Images.particle_map.data[im])
            #labels = self.Images.unique_particle_labels
            
            #if 0 in labels: labels = np.delete(labels, np.where(labels == 0))
            
            #labels -= 1 # Labelling commence at 1, but the index position starts at 0

            signal[im], lines = _utils._create_dummy_eds_spectra(labelled_image = self.Images.particle_map.data[im],
                                                                 elements = self.get_identified_elements(),
                                                                 label_concentrations = self.Particles.composition,#[:, labels],
                                                                 bkgr_idx = bkgr_idx,
                                                                 Erange = Erange,
                                                                 steps = steps)

            dissimilar_lines = list(set(LINES).symmetric_difference(set(lines)))

            if len(dissimilar_lines) > 0: 

                for line in dissimilar_lines: LINES.append(line)

        if regridify: 
            
            self.gridify_SEM_images(navigation_shape = nav_shape)

            signal = Signal1D(np.transpose(signal, (1,2,0,3)))

        else:

            signal = Signal1D(np.transpose(signal, (1,2,0)))
        
        # Bin the signal if the file size is too large
        if signal.data.nbytes / 1024**3 > 16:

            print(f'\nThe signal size ({signal.data.nbytes / 1024**3} Gb) is too large to do anything reasonable with... The signal will be downscaled.')
            
            signal = signal.rebin(scale = (1,2,2,1), dtype = "same") 

        if self.Images.is_gridified and stitch_signal: 

            print('Stitching the artificial signal')
            
            signal = Signal1D(_utils._reshape_artificial_eds_map(signal.data, nav_shape))

        signal.set_signal_type('EDS_SEM')

        signal.set_elements(self.get_identified_elements())

        signal.axes_manager[-1].name = 'Energy'
        
        signal.axes_manager[-1].units = 'keV'

        signal.axes_manager[-1].scale = steps

        return signal, LINES
                      
    
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #&&&&&&&&&&&&&&&&&&&& PARTICLE CLASSIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    def unclassify_particles(self, particle_arr):
        """Unclassify the particles defined by 'particles' array argument

        Parameters
        ----------
        particles 
            List of particles to unclassify. Note that the array must be of the
            same shape as the length of the classified particles.
            If particles is set to 'all', all particles will be unclassified.
        """
        ptype = type(particle_arr)

        if ptype == str: 
            
            if particle_arr == 'all': particle_arr = np.ones((self.number_of_particles), bool)
        
        if self._check_array_length(particle_arr): 
            
            # Set the particles of interest to not classified:
            self.is_classified[particle_arr] = False
            
            # Allocate corr. classification name 
            self.Particles.classes[particle_arr] = "Unclassified"
            
        else: print(f"Provided particle array of shape {np.shape(particle_arr)} do not match the total number of particles")
            
    def classify_particles(self, particle_arr, class_name):
        """Classify particles defined by argument 'particles'
        
        Parameters
        ----------
        particles
            array of particles to classify

        class_name 
            String that the particles will be labelled as
        """
        # Allocate a class name 
        if self._check_array_length(particle_arr):
            
            self.Particles.classes[particle_arr] = class_name
    
            # Update list of classified particles
            self.is_classified[particle_arr] = True

    def classify_noise(self, particle_arr):
        """
        Label specified particles as noise. I.e. particles defined by particle_arr
        is relabelled as Noise
        
        Parameters
        ----------
        particle_arr
            Array of particles to label as noise
        """
        
        # Allocate class name 
        
        if self._check_array_length(particle_arr):
            
            self.Particles.classes[particle_arr] = 'Noise'
    
            # Update list of classified particles
            self.is_classified[particle_arr] = True

    def tabulate_and_save_classified_particles_composition(self, path):
        """Create tables of the classified particles' composition.
        Note that the particle names will be the particles' label name.
        """
        
        path = str(path)

        if not os.path.isdir(directory_path):
            
            create_dir = input(f"The directory '{path}' does not exist. Create it? (y/[n])")

            if create_dir.upper() == 'Y':

                os.mkdir(path)

        classes = self.get_unique_particle_classes()

        for cl in classes:

            cl_arr = self.Particles.classes == cl
            
            utils.save_particles_property(
                self.Particles.composition[:,cl_arr],
                np.array(self.metadata.Particles.label_name)[cl_arr],
                self.get_identified_elements(),
                path = path,
                filename = f"{cl}_{self.metadata.chemical_unit}.txt"
            )

    

            





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UTILITIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
class utils:
    """ Particle analysis utilities"""

    def __init__(self, arg):

        self.csv_keys = pd.read_csv(arg).keys()

    def print_csv_metadata(csv_file):
        """Print the stored metadata in the particle analysis *.csv file

        Parameters
        ----------
        csv_file
            Either the csv filename used to load the data, or the pandas DataFrame object
        """
        
        if type(csv_file) == str: csv_file = pd.read_csv(str(csv_file))

        csv_keys = list(csv_file.keys())

        project_name = csv_keys[1]

        first_col, second_col = csv_file['Project name'], csv_file[project_name]

        # second_col[~first_col] is to avoid a vertical shift
        first_col, second_col = first_col[~first_col.isnull()], second_col[~first_col.isnull()] 

        # Replace nan == empty cell with empty string
        second_col[second_col.isnull()] = ''
        
        for first, second in zip(first_col, second_col): print(f"{first:35}{second:<20}")


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%% PRINTING PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    def print_particles_property(data, header, label = '', return_table = False):
        """Print selected particle's property like chemical composition or geometry
        
        Parameters
        ----------
        data
            Data to be printed. The data is expected to fit the shape (len(header), len(label))
        label
            List of labels : will be printed at the left of each row 
        header
            List of headers : will be printed at the top of each column

        Returns
        -------

        Example:
        >>>print_particles_property(data = (n,m) array of data, like (num elements, num particles)
                                 label = (n,) list of labels, like class names
                                 header = (m,) list of headers, like list of elements
        >>>
        """ 

        data_shape = data.shape

        if len(data_shape) <= 2:
            
            if type(label) == str: 
                
                if label == '': label = np.arange(0, data_shape[1])
        
            if data_shape[0] == len(header) and data_shape[1] == len(label):
            
                table = _utils._get_table(np.round(data, decimals = 2), label)

                if return_table: return table #tabulate(table, header, tablefmt="pretty") 
                
                else: print(tabulate(table, header, tablefmt="pretty"))

            else: print(f"The data shape ({data_shape}) doesn't fit the header ({len(header)}) and/or label ({len(label)}) shape(s)")
        
        else: print(f"Data of shape {data_shape} doesn't fit the table.")

    def save_particles_property(data, header, label = '', path = '', filename = 'tabulates.txt'):
        """Save the tabulated data in folder tabulates
        
        Parameters
        ----------
        data, label, header: see print_particles_property

        path
            String path to store the tabulated data. 

        filename
            Name of file being saved

        Note! The data will be stored in a filename called tabulates.txt.
        """

        folder = path

        save_results = False

        if folder[-1] != '\\' or folder[-1] != '/': folder += '\\'

        if os.path.exists(folder): save_results = True

        else: 
            
            ans = input(f"Couldn't find path {path}.\nCreate path? (y/[n])")

            if ans.upper() == 'Y' or ans == '':

                print(f"Creating path {path}")
                
                os.mkdir(path)

                save_results = True

        if save_results:

            if header == '': header = np.arange(0, np.shape(data)[1])

            table = _utils._get_table(data, label)

            filename = f"{os.path.splitext(filename)[0]}.txt"
            
            with open(folder + filename, 'w') as f:  f.write(tabulate(table, header, tablefmt="pretty"))


    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IMAGE MANIPULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def stitch_images(images, 
                      navigation_shape, 
                      horisontal_direction = 'r2l', 
                      vertical_direction = 't2b'):
        """Stitch an array of images into a 2D image. 

        Parameters
        ----------
        images
            Array with images to stitch. Expected shape: 4D (2 navigation, 2 image) dimensions
        navigation_shape 
            list or tuple of the concatenated image shape. 
            Note that the shape should reflect the numpy convention (i.e. 20 images -> 4x5 => shape = (5,4))
        vertical_direction
            Vertical direction at which the images are to be stitched.
            Allowed arguments: l2r and r2l (default: r2l: right to left)
        horisonal_direction
            Horisontal direction at which the images are to be stitched 
            Allowed arguments: t2b and b2t (default: t2b: top to bottom)

        Returns
        -------
        stitched_image 
            Concatenated images as a numpy array
        """
        if not _errors._check_for_numpy_ndarray(images): 
            
            raise TypeError(f'Argument data_array of type {type(images)} is not a numpy ndarray')
        
        if not len(navigation_shape) == 2: 

            if len(navigation_shape) == 1: navigation_shape = (1, navigation_shape[0])
            
            else: raise ValueError(f"Provided image shape {navigation_shape} is invalid. Provide a two dimensional shape.")

        if images.shape[:2] == navigation_shape:

            images = _image_utils._gridify_4D_array_to_3D(images)

        return _image_utils._stitch_images(images, 
                                      navigation_shape, 
                                      horisontal_direction = horisontal_direction, 
                                      vertical_direction = vertical_direction)

    def stitch_rgb_phase_map(phase_map, nav_shape):
        """Stitch a 4D or 5D array with 3 rgb channels into a single image of shape 3: (x,y,channels).

        Parameters
        ----------

        Returns
        -------
        
        Example
        -------
        """
        if not _errors._check_for_numpy_ndarray(phase_map): raise TypeError('Provided phase map must be a numpy ndarray')

        if len(nav_shape) != 2: raise ValueError(f"Provided navigation shape {nav_shape} is not valid.")
        
        phase_map_s = phase_map.copy()

        # 20*20 != 20
        if np.prod(nav_shape) == np.prod(phase_map.shape[:2]) and len(phase_map) > 4: 

            phase_map_s = _image_utils._image_utils._gridify_ND_array_to_nD(phase_map_s)

        return _image_utils._stitch_images(phase_map_s, shape = nav_shape)
        
    
    def greyscale_to_rgba(grey_image, dtype_out = np.float16):
        """Return a grey-scale array as a rgb equivalent
        
        Parameters
        ----------
        grey_image
            2D grey scale image array 
    
        Returns
        -------
            2D image array with RGBA channels (RGB is normalised to be in the range 0,1)
        """
        fac = np.max(grey_image)
        
        if not _errors._check_for_numpy_ndarray(grey_image): raise TypeError(f"Input image is not a grey scale.")
        
        img = np.expand_dims(grey_image, axis = -1).astype(dtype_out)
        
        return np.concatenate((img / fac, img / fac, img / fac,
                               np.full_like(img, 1)), # alpha channel
                               axis = -1)

    def greyscale_to_rgb(grey_image, 
                         in_range=(0, 255),
                         dtype_out = np.float16):
        """Return a grey-scale array as a normalised rgb equivalent (intensity range: 0,1)
        
        Parameters
        ----------
        grey_image
            2D grey scale image array 
        in_range
            tuple of in range intensity values that is givne to the rescale_intensity function
        dtype_out
            Datatype to return            
    
        Returns
        -------
            2D image array with RGBA channels (RGB is normalised to be in the range 0,1)
        """
                
        if not _errors._check_for_numpy_ndarray(grey_image): raise TypeError(f"Input image is not a grey scale.")
        
        from skimage import color
        from skimage.exposure import rescale_intensity

        grey_im = color.gray2rgb(grey_image) # Intensity values unchanged
        
        return (rescale_intensity(1.0 * grey_im, in_range = in_range)).astype(np.float32)

    def gridify_3D_array_to_4D(arr, nav_shape):
        """Gridify the 3D array to 4D. Nav_shape defines the number of images in the different directions.

        Parameters
        ----------
        arr
            numpy.ndarray of shape (3,)
        nav_shape
            Navigation shape to shape the array into

        Example
        ------
        >>> import particle_analysis as pa
        >>> import numpy as np
        >>> img = np.asarray([[[0]*4]*4]*4)
        >>> img.shape
        (4,4,4)
        >>> img = pa.gridify_3D_array_to_4D(arr, nav_shape = (2,2))
        >>> img.shape
        (2,2,4,4)
        """
        if len(arr.shape) != 3: raise TypeError(f"Array shape {arr.shape} is not expected.")

        if len(nav_shape) != 2: raise TypeError(f"Navigation shape is not valid. Provide a 2-integer list")

        return _image_utils._gridify_3D_array_to_4D(arr, nav_shape + arr.shape[-2:])
        
    def plot_rgb_map_with_colorbar(array, 
                                   colours, 
                                   background_colour = 'whitesmoke',
                                   return_fig = False):
        from matplotlib import colors
        import matplotlib.pyplot as plt
        
        auto_colouring  = False

        colour_type = type(colours)

        num_colors = len(colours)
        
        if colours is not None: 

            unique_classes = list(colours.keys())

            if colour_type == dict: colours = [colours[cl] for cl in unique_classes]

        else: auto_colouring = True

        # Create a unique color map
        if auto_colouring: 

            if len(colours) < 11: 
                
                print('Colouring according to tableau colors')
                
                colours = [col for col in list(colors.TABLEAU_COLORS.keys())[:len(unique_classes)]]
            
            else: 
                
                print('Generating random colours')
                # Alternatively, use: colors.CSS4_COLORS
                colours = [_utils._generate_random_rgb_color() for i in range(len(unique_classes))]

        colours.insert(0, colors.to_rgb(background_colour))

        unique_classes.insert(0, 'Matrix')
        
        phase_vals = np.arange(num_colors + 1)
        
        cmap = colors.ListedColormap(colours)
        
        norm = colors.BoundaryNorm(np.arange(-0.5, phase_vals.max() + 1.5, 1), cmap.N)

        # Plotting
        fig, ax = plt.subplots()
        cax = ax.imshow(array, cmap = cmap, norm = norm)
        # Add a colorbar with a label
        cbar = fig.colorbar(cax, ticks = phase_vals)
        
        if colour_type == dict: cbar.ax.set_yticklabels(unique_classes) 
            
        plt.axis('off')
        plt.show()

        if return_fig: return fig
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARTICLE CHEMISTRY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def plot_data_distribution(data_arr,
                               unit = '',
                               x_tick = '',
                               colors = [np.asarray([21,106,163]) / 255, # violin
                                         np.asarray([175,204,184])#191,187,152]) 
                                                         / 255, # boxplot
                                         np.asarray([206,156,168]) / 255], # scatter
                               return_fig = False):
        """Plot a box + violin + scatter plot of data_arr

        Parameters
        ----------
        data_arr
            np.ndarray of shape (N,)
        """
        import matplotlib.pyplot as plt

        # --- Temporary change using rc_context() ---
        with plt.rc_context({'lines.linewidth': 3, 'font.size' : 16}):
        
            fig, axs = plt.subplots(figsize=(10,10))
    
            boxprops = dict(linestyle='-', linewidth=2, color='k')
            
            medianprops = dict(linestyle='-', linewidth=2, color='k')
            
            bplot = axs.boxplot(data_arr, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
            
            vp1 = axs.violinplot(data_arr, showmeans=False, showmedians=False, side = 'high', showextrema=False)

            for pc in vp1['bodies']:
                pc.set_facecolor(colors[0])
                pc.set_edgecolor('black')
            
            axs.set_ylabel(unit)
            
            for patch, color in zip(bplot['boxes'], [colors[1]]): 
                patch.set_alpha(0.8)
                patch.set_facecolor(color)
            
            x_arr = np.random.randint(low = 95, high = 105, size = len(data_arr)) / 100
            
            scatter = axs.scatter(x_arr, data_arr, color=colors[2], marker='o', zorder=5, alpha=.35)
            
            # Legends
            axs.legend([bplot["boxes"][0], vp1['bodies'][0], scatter], 
                       ['Box plot', 'Violin plot', 'Data pts.'], 
                       loc='upper right')

            plt.xticks([1], [x_tick])
    
            plt.show()

        if return_fig: return fig

    def get_label_colourmap(list_of_colours : list | None = None):
        """Create a colourmap for labels.

        Parameters
        ----------
        list_of_colours
            List of pyplot colour names
        num_colours
            Number of colours in the colourmap

        returns
        -------
        colour map
            A matplotlib.color ListedColormap
        """
        return _colouring.get_discrete_colour_map(list_of_colours)

    def get_navigator_colours(image):
        """Create a navigator rgb map
        """
        return _colouring.get_rgb_navigator(image)

    def get_greek_letters(letter : str):
        """Return the ¨code representing greek letters for nice printing/name setting"""
        


            

        