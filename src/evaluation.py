from tqdm import tqdm
import pandas as pd
import numpy as np

import settings


def array_to_string(a):

    """
    Flatten given array and convert it to a string with semicolon delimiters

    Parameters
    ----------
    a: np.ndarray
        N-dimensional array

    Returns
    -------
    s: string
        String form of the given array
    """

    s = ';'.join([str(x) for x in a.reshape(-1)])

    return s


def string_to_array(s):

    """
    Convert semicolon delimited string to an array

    Parameters
    ----------
    s: string
        String form of the array

    Returns
    -------
    a: np.ndarray
        N-dimensional array
    """

    a = np.array(s.split(';')).astype(np.float64)

    return a


def rotation_matrix_to_quaternion(rotation_matrix):

    """
    Convert rotation matrix to quaternion

    Parameters
    ----------
    rotation_matrix: numpy.ndarray of shape (3, 3)
        Array of directions of the world-axes in camera coordinates

    Returns
    -------
    quaternion: numpy.ndarray of shape (4)
        Array of quaternion
    """

    r00 = rotation_matrix[0, 0]
    r01 = rotation_matrix[0, 1]
    r02 = rotation_matrix[0, 2]
    r10 = rotation_matrix[1, 0]
    r11 = rotation_matrix[1, 1]
    r12 = rotation_matrix[1, 2]
    r20 = rotation_matrix[2, 0]
    r21 = rotation_matrix[2, 1]
    r22 = rotation_matrix[2, 2]

    k = np.array([
        [r00 - r11 - r22, 0.0, 0.0, 0.0],
        [r01 + r10, r11 - r00 - r22, 0.0, 0.0],
        [r02 + r20, r12 + r21, r22 - r00 - r11, 0.0],
        [r21 - r12, r02 - r20, r10 - r01, r00 + r11 + r22]
    ])
    k /= 3.0

    # Quaternion is the eigenvector of k that corresponds to the largest eigenvalue
    w, v = np.linalg.eigh(k)
    quaternion = v[[3, 0, 1, 2], np.argmax(w)]

    if quaternion[0] < 0:
        np.negative(quaternion, quaternion)

    return quaternion


def pose_difference(r1, t1, r2, t2):

    """
    Calculate relative pose difference from given rotation matrices and translation vectors

    Parameters
    ----------
    r1: numpy.ndarray of shape (3, 3)
        First rotation matrix

    t1: numpy.ndarray of shape (3)
        First translation vector

    r2: numpy.ndarray of shape (3, 3)
        Second rotation matrix

    t2: numpy.ndarray of shape (3)
        Second translation vector

    Returns
    -------
    rotation_difference: float
        Rotation difference in terms of degrees from the first image

    translation_difference: float
        Translation difference in terms of meters from the first image
    """

    rotation_difference = np.dot(r2, r1.T)
    translation_difference = t2 - np.dot(rotation_difference, t1)

    return rotation_difference, translation_difference


def rotation_and_translation_error(q_ground_truth, t_ground_truth, q_prediction, t_prediction, epsilon=1e-15):

    """
    Calculate rotation and translation error

    Parameters
    ----------
    q_ground_truth: numpy.ndarray of shape (4)
        Array of quaternion derived from ground truth rotation matrix

    t_ground_truth: numpy.ndarray of shape (3)
        Array of ground truth translation vector

    q_prediction: numpy.ndarray of shape (4)
        Array of quaternion derived from estimated rotation matrix

    t_prediction: numpy.ndarray of shape (3)
        Array of estimated translation vector

    epsilon: float
        A small number for preventing zero division

    Returns
    -------
    rotation_error: float
        Rotation error in terms of degrees

    translation_error: float
        Translation error in terms of meters
    """

    q_ground_truth_norm = q_ground_truth / (np.linalg.norm(q_ground_truth) + epsilon)
    q_prediction_norm = q_prediction / (np.linalg.norm(q_prediction) + epsilon)
    loss_q = np.maximum(epsilon, (1.0 - np.sum(q_prediction_norm * q_ground_truth_norm) ** 2))

    rotation_error = np.degrees(np.arccos(1 - (2 * loss_q)))

    scaling_factor = np.linalg.norm(t_ground_truth)
    t_prediction = scaling_factor * (t_prediction / (np.linalg.norm(t_prediction) + epsilon))
    translation_error = min(
        np.linalg.norm(t_ground_truth - t_prediction),
        np.linalg.norm(t_ground_truth + t_prediction)
    )

    return rotation_error, translation_error


def mean_average_accuracy(rotation_errors, translation_errors, rotation_error_thresholds, translation_error_thresholds):

    """
    Calculate mean average accuracies over a set of thresholds for rotation and translation

    Parameters
    ----------
    rotation_errors: list of shape (n_pairs)
        List of rotation errors

    translation_errors: list of shape (n_pairs)
        List of translation errors

    rotation_error_thresholds: numpy.ndarray of shape (10)
        Array of rotation error thresholds

    translation_error_thresholds: numpy.ndarray of shape (10)
        Array of translation error thresholds

    Returns
    -------
    maa: float
        Mean average accuracy calculated on both rotation and translation errors

    rotation_maa: float
        Mean average accuracy calculated on rotation errors

    translation_maa: float
        Mean average accuracy calculated on translation errors
    """

    accuracies, rotation_accuracies, translation_accuracies = [], [], []

    for rotation_error_threshold, translation_error_threshold in zip(rotation_error_thresholds, translation_error_thresholds):

        # Calculate whether the errors are less than specified thresholds or not
        rotation_accuracy = (rotation_errors <= rotation_error_threshold)
        translation_accuracy = (translation_errors <= translation_error_threshold)
        accuracy = rotation_accuracy & translation_accuracy

        accuracies.append(accuracy.astype(np.float32).mean())
        rotation_accuracies.append(rotation_accuracy.astype(np.float32).mean())
        translation_accuracies.append(translation_accuracy.astype(np.float32).mean())

    maa = np.array(accuracies).mean()
    rotation_maa = np.array(rotation_accuracies).mean()
    translation_maa = np.array(translation_accuracies).mean()

    return maa, rotation_maa, translation_maa


def evaluate(df, verbose=False):

    """
    Calculate mean average accuracies over a set of thresholds for rotation and translation

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with dataset, scene, rotation_matrix, translation_vector, rotation_matrix_prediction and translation_vector_prediction columns

    verbose: bool
        Whether to print scores or not

    Returns
    -------
    df_scores: pandas.DataFrame
        Dataframe of scores
    """

    rotation_error_thresholds = {
        **{('haiper', scene): np.linspace(1, 10, 10) for scene in ['bike', 'chairs', 'fountain']},
        **{('heritage', scene): np.linspace(1, 10, 10) for scene in ['cyprus', 'dioscuri']},
        **{('heritage', 'wall'): np.linspace(0.2, 10, 10)},
        **{('urban', 'kyiv-puppet-theater'): np.linspace(1, 10, 10)},
    }
    translation_error_thresholds = {
        **{('haiper', scene): np.geomspace(0.05, 0.5, 10) for scene in ['bike', 'chairs', 'fountain']},
        **{('heritage', scene): np.geomspace(0.1, 2, 10) for scene in ['cyprus', 'dioscuri']},
        **{('heritage', 'wall'): np.geomspace(0.05, 1, 10)},
        **{('urban', 'kyiv-puppet-theater'): np.geomspace(0.5, 5, 10)},
    }
    df_scores = pd.DataFrame(columns=['dataset', 'scene', 'image_pairs', 'maa', 'rotation_maa', 'translation_maa'])

    for (dataset, scene), df_scene in tqdm(df.groupby(['dataset', 'scene'])):

        scene_rotation_errors = []
        scene_translation_errors = []

        for i in range(df_scene.shape[0]):
            for j in range(i + 1, df_scene.shape[0]):

                rotation_matrix_difference_ground_truth, translation_vector_difference_ground_truth = pose_difference(
                    r1=string_to_array((df_scene.iloc[i]['rotation_matrix'])).reshape(3, 3),
                    t1=string_to_array((df_scene.iloc[i]['translation_vector'])),
                    r2=string_to_array((df_scene.iloc[j]['rotation_matrix'])).reshape(3, 3),
                    t2=string_to_array((df_scene.iloc[j]['translation_vector'])),
                )
                quaternion_ground_truth = rotation_matrix_to_quaternion(rotation_matrix=rotation_matrix_difference_ground_truth)

                rotation_matrix_difference_prediction, translation_vector_difference_prediction = pose_difference(
                    r1=string_to_array((df_scene.iloc[i]['rotation_matrix_prediction'])).reshape(3, 3),
                    t1=string_to_array((df_scene.iloc[i]['translation_vector_prediction'])),
                    r2=string_to_array((df_scene.iloc[j]['rotation_matrix_prediction'])).reshape(3, 3),
                    t2=string_to_array((df_scene.iloc[j]['translation_vector_prediction'])),
                )
                quaternion_prediction = rotation_matrix_to_quaternion(rotation_matrix=rotation_matrix_difference_prediction)

                rotation_error, translation_error = rotation_and_translation_error(
                    q_ground_truth=quaternion_ground_truth,
                    t_ground_truth=translation_vector_difference_ground_truth,
                    q_prediction=quaternion_prediction,
                    t_prediction=translation_vector_difference_prediction,
                    epsilon=1e-15
                )
                scene_rotation_errors.append(rotation_error)
                scene_translation_errors.append(translation_error)

        scene_maa, scene_rotation_maa, scene_translation_maa = mean_average_accuracy(
            rotation_errors=scene_rotation_errors,
            translation_errors=scene_translation_errors,
            rotation_error_thresholds=rotation_error_thresholds[(dataset, scene)],
            translation_error_thresholds=translation_error_thresholds[(dataset, scene)]
        )

        if verbose:
            settings.logger.info(
                f'''
                Dataset: {dataset} - Scene: {scene}
                Number of image pairs: {len(scene_rotation_errors)}
                mAA: {scene_maa:.6f} - rotation mAA: {scene_rotation_maa:.6f} - translation mAA: {scene_translation_maa:.6f}
                '''
            )

        df_scores = pd.concat((
            df_scores,
            pd.DataFrame(
                data=[[scene, dataset, len(scene_rotation_errors), scene_maa, scene_rotation_maa, scene_translation_maa]],
                columns=['dataset', 'scene', 'image_pairs', 'maa', 'rotation_maa', 'translation_maa']
            )
        ), axis=0)

    return df_scores
