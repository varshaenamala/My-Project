function h = confusionchart(varargin)


narginchk(1, Inf);

try
    % Check input args are valid, throw error here if they're not.
    [parent, model, cellOfKeyValuePairs] = iParseInput(varargin{:});
    
    % If the user hasn't specified an 'OuterPosition' value, we try and
    % replace the existing axes (if one exists). If they have specified a
    % 'Position', 'InnerPosition' or 'OuterPosition' value, we make a new
    % chart and leave any existing axes alone.
    if ~iHasPositionArg(cellOfKeyValuePairs)
        constructorFcn = @(varargin)(mlearnlib.graphics.chart.ConfusionMatrixChart(...
            varargin{:}, 'Model', model, cellOfKeyValuePairs{:}));
        
        % If the user hasn't defined a parent, parent will be empty. This
        % will be handled correctly by prepareCoordinateSystem.
        cm = matlab.graphics.internal.prepareCoordinateSystem('mlearnlib.graphics.chart.ConfusionMatrixChart', parent, constructorFcn);
    else
        % If the user hasn't defined a parent, we need to get one now.
        if isempty(parent)
           parent = gcf(); 
        end

        cm = mlearnlib.graphics.chart.ConfusionMatrixChart('Parent', parent, 'Model', model, cellOfKeyValuePairs{:});
    end
    
    fig = ancestor(cm, 'Figure');
    if isscalar(fig)
        fig.CurrentAxes = cm;
    end
catch e
    throw(e);
end

% Prevent outputs when not assigning to variable.
if nargout > 0
    h = cm; 
end

end

function [parent, model, cellOfKeyValuePairs] = iParseInput(varargin)
% Parse input to standard form. We return the parent separately (even
% though it's included in the args) so we can use it in
% matlab.graphics.internal.prepareCoordinateSystem.

parentValidator = mlearnlib.internal.confusionmatrixchart.input.ParentValidator();
syntaxInterpreter = mlearnlib.internal.confusionmatrixchart.factories.SyntaxInterpreter();

parser = mlearnlib.internal.confusionmatrixchart.input.InputParser(parentValidator, syntaxInterpreter);

[parent, model, cellOfKeyValuePairs] = parser.parse(varargin{:});
end

function tf = iHasPositionArg(cellOfKeyValuePairs)
% Returns true if the name-value pairs contain a 'Position' name of some
% sort.

propNames = cellOfKeyValuePairs(1:2:end);
tf = any(strcmpi(propNames, 'Position')) || ...
    any(strcmpi(propNames, 'InnerPosition')) || ...
    any(strcmpi(propNames, 'OuterPosition'));
end