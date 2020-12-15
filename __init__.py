import logging
import re

from googleapiclient.errors import HttpError
from airflow.contrib.hooks.gcp_mlengine_hook import MLEngineHook
from airflow.exceptions import AirflowException
from airflow.operators import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.plugins_manager import AirflowPlugin

log = logging.getLogger(__name__)

def _normalize_mlengine_job_id(job_id):
    """
    Replaces invalid MLEngine job_id characters with '_'.

    This also adds a leading 'z' in case job_id starts with an invalid
    character.

    Args:
        job_id: A job_id str that may have invalid characters.

    Returns:
        A valid job_id representation.
    """

    # Add a prefix when a job_id starts with a digit or a template
    match = re.search(r'\d|\{{2}', job_id)
    if match and match.start() == 0:
        job = 'z_{}'.format(job_id)
    else:
        job = job_id

    # Clean up 'bad' characters except templates
    tracker = 0
    cleansed_job_id = ''
    for m in re.finditer(r'\{{2}.+?\}{2}', job):
        cleansed_job_id += re.sub(r'[^0-9a-zA-Z]+', '_',
                                  job[tracker:m.start()])
        cleansed_job_id += job[m.start():m.end()]
        tracker = m.end()

    # Clean up last substring or the full string if no templates
    cleansed_job_id += re.sub(r'[^0-9a-zA-Z]+', '_', job[tracker:])

    return cleansed_job_id

class MLEngineTrainingOperatorV2(BaseOperator):
    """
    Operator for launching a MLEngine training job.

    :param project_id: The Google Cloud project name within which MLEngine
        training job should run (templated).
    :type project_id: str

    :param job_id: A unique templated id for the submitted Google MLEngine
        training job. (templated)
    :type job_id: str

    :param package_uris: A list of package locations for MLEngine training job,
        which should include the main training program + any additional
        dependencies. (templated)
    :type package_uris: str

    :param training_python_module: The Python module name to run within MLEngine
        training job after installing 'package_uris' packages. (templated)
    :type training_python_module: str

    :param training_args: A list of templated command line arguments to pass to
        the MLEngine training program. (templated)
    :type training_args: str

    :param region: The Google Compute Engine region to run the MLEngine training
        job in (templated).
    :type region: str

    :param scale_tier: Resource tier for MLEngine training job. (templated)
    :type scale_tier: str

    :param master_type: Cloud ML Engine machine name.
        Must be set when scale_tier is CUSTOM. (templated)
    :type master_type: str

    :param runtime_version: The Google Cloud ML runtime version to use for
        training. (templated)
    :type runtime_version: str

    :param python_version: The version of Python used in training. (templated)
    :type python_version: str

    :param job_dir: A Google Cloud Storage path in which to store training
        outputs and other data needed for training. (templated)
    :type job_dir: str

    :param gcp_conn_id: The connection ID to use when fetching connection info.
    :type gcp_conn_id: str

    :param delegate_to: The account to impersonate, if any.
        For this to work, the service account making the request must have
        domain-wide delegation enabled.
    :type delegate_to: str

    :param mode: Can be one of 'DRY_RUN'/'CLOUD'. In 'DRY_RUN' mode, no real
        training job will be launched, but the MLEngine training job request
        will be printed out. In 'CLOUD' mode, a real MLEngine training job
        creation request will be issued.
    :type mode: str
    """
    template_fields = [
        '_project_id',
        '_job_id',
        '_package_uris',
        '_training_python_module',
        '_training_args',
        '_region',
        '_scale_tier',
        '_master_type',
        '_master_image_uri',
        '_runtime_version',
        '_python_version',
        '_job_dir'
    ]

    @apply_defaults
    def __init__(self,
                 project_id,
                 job_id,
                 package_uris,
                 training_python_module,
                 training_args,
                 region,
                 scale_tier=None,
                 master_type=None,
                 master_image_uri=None,
                 runtime_version=None,
                 python_version=None,
                 job_dir=None,
                 gcp_conn_id='google_cloud_default',
                 delegate_to=None,
                 mode='PRODUCTION',
                 *args,
                 **kwargs):
        super(MLEngineTrainingOperatorV2, self).__init__(*args, **kwargs)
        self._project_id = project_id
        self._job_id = job_id
        self._package_uris = package_uris
        self._training_python_module = training_python_module
        self._training_args = training_args
        self._region = region
        self._scale_tier = scale_tier
        self._master_type = master_type
        self._master_image_uri = master_image_uri
        self._runtime_version = runtime_version
        self._python_version = python_version
        self._job_dir = job_dir
        self._gcp_conn_id = gcp_conn_id
        self._delegate_to = delegate_to
        self._mode = mode

        if not self._project_id:
            raise AirflowException('Google Cloud project id is required.')
        if not self._job_id:
            raise AirflowException(
                'An unique job id is required for Google MLEngine training '
                'job.')
        if not package_uris:
            raise AirflowException(
                'At least one python package is required for MLEngine '
                'Training job.')
        if not training_python_module:
            raise AirflowException(
                'Python module name to run after installing required '
                'packages is required.')
        if not self._region:
            raise AirflowException('Google Compute Engine region is required.')
        if self._scale_tier is not None and self._scale_tier.upper() == "CUSTOM" and not self._master_type:
            raise AirflowException(
                'master_type must be set when scale_tier is CUSTOM')

    def execute(self, context):
        job_id = _normalize_mlengine_job_id(self._job_id)
        training_request = {
            'jobId': job_id,
            'trainingInput': {
                'scaleTier': self._scale_tier,
                'packageUris': self._package_uris,
                'pythonModule': self._training_python_module,
                'region': self._region,
                'args': self._training_args
            }
        }

        if self._runtime_version:
            training_request['trainingInput']['runtimeVersion'] = self._runtime_version

        if self._master_image_uri:
            training_request['trainingInput']['masterConfig'] = {}
            training_request['trainingInput']['masterConfig']['imageUri'] = self._master_image_uri

        if self._python_version:
            training_request['trainingInput']['pythonVersion'] = self._python_version

        if self._job_dir:
            training_request['trainingInput']['jobDir'] = self._job_dir

        if self._scale_tier is not None and self._scale_tier.upper() == "CUSTOM":
            training_request['trainingInput']['masterType'] = self._master_type

        if self._mode == 'DRY_RUN':
            self.log.info('In dry_run mode.')
            self.log.info('MLEngine Training job request is: %s',
                          training_request)
            return

        hook = MLEngineHook(
            gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)

        # Helper method to check if the existing job's training input is the
        # same as the request we get here.
        def check_existing_job(existing_job):
            return existing_job.get('trainingInput', None) == \
                training_request['trainingInput']

        try:
            finished_training_job = hook.create_job(
                self._project_id, training_request, check_existing_job)
        except HttpError:
            raise

        if finished_training_job['state'] != 'SUCCEEDED':
            self.log.error('MLEngine training job failed: %s',
                           str(finished_training_job))
            raise RuntimeError(finished_training_job['errorMessage'])


class MLEngineTrainingOperatorPlugin(AirflowPlugin):
    name = "MLEngineTrainingOperatorPlugin"
    hooks = []
    operators = [MLEngineTrainingOperatorV2]
    executors = []
    macros = []
    admin_views = []
    flask_blueprints = []
    menu_links = []
