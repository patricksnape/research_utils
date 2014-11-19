import os.path
import cPickle
import menpo.io as mio
from menpo.visualize import print_dynamic
import numpy as np
from mesh_processing import preprocess_image
from custom_importers import ABSImporter
from pathlib import Path


def load_frgc(session_id, recreate_meshes=False,
              output_base_path=Path('/vol/atlas/homes/pts08/'),
              input_base_path=Path('/vol/atlas/databases/frgc'),
              max_images=None):
    previously_pickled_path = output_base_path / 'frgc_{0}_68_cleaned.pkl'.format(session_id)
    abs_files_path = input_base_path / session_id / '*.abs'

    if not recreate_meshes and previously_pickled_path.exists():
        with open(str(previously_pickled_path)) as f:
            images = cPickle.load(f)
    else:
        # Add the custom ABS importer
        from menpo.io.input.extensions import image_types
        image_types['.abs'] = ABSImporter

        images = []
        for i, im in enumerate(mio.import_images(abs_files_path,
                                                 max_images=max_images,
                                                 verbose=True)):
            if im.n_landmark_groups > 0:
                preprocess_image(im)
                images.append(im)

        # Only dump the saved images if we loaded all of them!
        if max_images is None:
            with open(str(previously_pickled_path), 'wb') as f:
                cPickle.dump(images, f, protocol=2)

    return images


def load_basel_from_mat(recreate_meshes=False,
                        output_base_path='/vol/atlas/homes/pts08/',
                        input_base_path='/vol/atlas/pts08/basel/',
                        max_images=None):
    previously_pickled_path = os.path.join(output_base_path,
                                           'basel_python_68.pkl')
    mat_file_path = os.path.join(input_base_path, 'basel_68.mat')

    if not recreate_meshes and os.path.exists(previously_pickled_path):
        with open(previously_pickled_path) as f:
            images = cPickle.load(f)
    else:
        from scipy.io import loadmat
        from menpo.image import Image
        from menpo.shape import PointCloud

        basel_dataset = loadmat(mat_file_path)
        textures = basel_dataset['textures']
        shape = basel_dataset['shapes']
        landmarks = np.swapaxes(basel_dataset['landmarks'], 0, 1)

        N = max_images if max_images else landmarks.shape[2]

        all_images = []

        for i in xrange(N):
            # Change to the correct handedness
            shape[..., 1:3, i] *= -1
            shape_image = ShapeImage(shape[..., i],
                                     texture=Image(textures[..., i]))
            shape_image.landmarks['PTS'] = PointCloud(landmarks[..., i])
            shape_image.mesh.texture.landmarks['PTS'] = shape_image.landmarks['PTS']
            shape_image.constrain_mask_to_landmarks()
            shape_image.rebuild_mesh()
            all_images.append(shape_image)
            print_dynamic('Image {0} of {1}'.format(i + 1, N))

        print('\n')
        images = [im for im in all_images if im.n_landmark_groups == 1]
        print('{0}% of the images had landmarks'.format(
            (float(len(images)) / len(all_images)) * 100))
        with open(previously_pickled_path, 'wb') as f:
            cPickle.dump(images, f, protocol=2)

    return images